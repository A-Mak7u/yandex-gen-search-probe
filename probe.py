#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yandex Generative Search Probe (RAG Reverse-Engineering Tool)
------------------------------------------------------------
Аналитический инструмент для исследования контура генеративного поиска Яндекса
(Search API v2 /v2/gen/search): Query Rewriting, разделение retrieved vs used sources,
оценка стабильности выдачи (Jaccard similarity) и доменной энтропии.

Зависимости: стандартная библиотека Python 3.9+ (zero third-party dependencies).
Лицензия: MIT.
Автор: Mak7u (https://github.com/A-Mak7u)
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse

API_ENDPOINT = "https://searchapi.api.cloud.yandex.net/v2/gen/search"
DEFAULT_PRICE_PER_1K = 5.08  # Рублей за 1 000 запросов с НДС
BOLD_PATTERN = re.compile(r'\*\*(.+?)\*\*')


def parse_gen_response(raw_text: str) -> Dict[str, Any]:
    """Разбирает потоковый JSONL или JSON-массив ответа эндпоинта /v2/gen/search."""
    raw_text = raw_text.strip()
    if not raw_text:
        return {"error": "Empty response body"}
    
    items = json.loads(raw_text) if raw_text.startswith('[') else [
        json.loads(x) for x in raw_text.splitlines() if x.strip()
    ]
    last = items[-1] if items else {}
    sources = last.get("sources", [])
    
    return {
        "answer": (last.get("message") or {}).get("content", ""),
        "used_sources": [s.get("url", "") for s in sources if s.get("used")],
        "other_sources": [s.get("url", "") for s in sources if not s.get("used")],
        "search_queries": [q.get("text", "") for q in last.get("searchQueries", [])],
        "rejected": bool(last.get("isAnswerRejected")),
        "bullet": bool(last.get("isBulletAnswer")),
    }


def query_api(question: str, api_key: str, folder_id: str, timeout: int = 60) -> Dict[str, Any]:
    """Выполняет регламентный POST-запрос к генеративному эндпоинту."""
    payload = {
        "messages": [{"content": question, "role": "ROLE_USER"}],
        "folderId": folder_id,
        "searchType": "SEARCH_TYPE_RU"
    }
    req = urllib.request.Request(
        API_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "yandex-gen-search-probe/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = parse_gen_response(resp.read().decode("utf-8"))
            data["error"] = ""
            return data
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", "ignore")[:300]
        return {
            "answer": "",
            "used_sources": [],
            "other_sources": [],
            "search_queries": [],
            "rejected": False,
            "bullet": False,
            "error": f"HTTP {e.code}: {err_msg}"
        }


def extract_domain(url: str) -> str:
    """Извлекает чистый домен из URL без www."""
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.").lower()


def normalize_text(text: str) -> str:
    """Нормализует текст: убирает кавычки, приводит ё -> е и к нижнему регистру."""
    return re.sub(r'[«»"*]', '', text.lower().replace('ё', 'е'))


def contains_brand(text: str, aliases: List[str]) -> bool:
    """Проверяет наличие упоминания бренда с учетом масок и падежных окончаний."""
    norm = normalize_text(text)
    for a in aliases:
        pattern = r'(?<![\w.])' + re.escape(a).replace(r'\*', r'\w*') + r'(?![\w])'
        if re.search(pattern, norm):
            return True
    return False


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Загружает JSONL датасет."""
    p = Path(file_path)
    if not p.exists():
        sys.exit(f"Ошибка: Файл не найден: {file_path}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_report(rows: List[Dict[str, Any]], brands_spec: str):
    """Анализирует цитируемость сайтов и упоминаемость брендов в ответах."""
    ok_rows = [r for r in rows if not r.get("error")]
    total = len(ok_rows)
    print(f"Всего валидных ответов: {total} (ошибок: {len(rows) - total})")
    print("-" * 75)
    print(f"{'Бренд':<20} | {'Упоминание в тексте':<22} | {'Сайт в used_sources':<22}")
    print("-" * 75)
    
    brand_entries = []
    for item in brands_spec.split(";"):
        parts = [p.strip() for p in item.split("|") if p.strip()]
        if parts:
            brand_entries.append((parts[0], [p.lower().replace('ё', 'е') for p in parts]))
            
    for name, aliases in brand_entries:
        named_count = sum(contains_brand(r.get("answer", ""), aliases) for r in ok_rows)
        domains = [a for a in aliases if "." in a and " " not in a]
        cited_count = sum(
            any(extract_domain(u).endswith(d) for u in r.get("used_sources", []) for d in domains)
            for r in ok_rows
        )
        named_pct = (named_count / total * 100) if total else 0.0
        cited_pct = (cited_count / total * 100) if total else 0.0
        print(f"{name:<20} | {named_count:3d} ({named_pct:5.1f}%)         | {cited_count:3d} ({cited_pct:5.1f}%)")
    print("-" * 75)


def cmd_jaccard(rows: List[Dict[str, Any]]):
    """Рассчитывает Jaccard Stability Index между прогонами одних и тех же вопросов."""
    by_q: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        if not r.get("error"):
            by_q.setdefault(r.get("question", ""), {})[r.get("run", 1)] = r
            
    similarities = []
    identical_count = 0
    
    for q, runs in by_q.items():
        if len(runs) >= 2:
            keys = sorted(runs.keys())
            run1, run2 = runs[keys[0]], runs[keys[1]]
            dom1 = {extract_domain(u) for u in run1.get("used_sources", [])}
            dom2 = {extract_domain(u) for u in run2.get("used_sources", [])}
            
            if not dom1 and not dom2:
                j = 1.0
            else:
                j = len(dom1 & dom2) / len(dom1 | dom2) if (dom1 | dom2) else 1.0
                
            similarities.append(j)
            if dom1 == dom2:
                identical_count += 1
                
    if not similarities:
        print("Внимание: В датасете нет парных прогонов (run=1, run=2) для расчета стабильности.")
        return
        
    print("=== АНАЛИЗ СТАБИЛЬНОСТИ ИСТОЧНИКОВ (JACCARD SIMILARITY) ===")
    print(f"Сравниваемых пар вопросов: {len(similarities)}")
    print(f"Полное совпадение источников (J = 1.0): {identical_count} ({identical_count / len(similarities) * 100:.1f}%)")
    print(f"Медиана сходства: {statistics.median(similarities):.4f}")
    print(f"Среднее сходство: {statistics.mean(similarities):.4f}")
    print(f"Минимум / Максимум: {min(similarities):.4f} / {max(similarities):.4f}")
    print(f"Нулевое пересечение (J = 0.0): {sum(1 for s in similarities if s == 0.0)}")


def cmd_entropy(rows: List[Dict[str, Any]], top_k: int = 15):
    """Анализирует распределение источников, концентрацию ТОП-10 и энтропию Шеннона."""
    ok_rows = [r for r in rows if not r.get("error")]
    domains = [extract_domain(u) for r in ok_rows for u in r.get("used_sources", []) if u]
    counter = Counter(domains)
    total_links = sum(counter.values())
    unique_domains = len(counter)
    
    top_10_links = sum(c for _, c in counter.most_common(10))
    top_10_share = (top_10_links / total_links * 100) if total_links else 0.0
    
    # Шенноновская энтропия
    probs = [c / total_links for c in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(unique_domains) if unique_domains > 1 else 1.0
    
    print("=== АНАЛИЗ РАСПРЕДЕЛЕНИЯ И ЭНТРОПИИ ИСТОЧНИКОВ ===")
    print(f"Всего ссылок на источники: {total_links}")
    print(f"Уникальных доменов: {unique_domains}")
    print(f"Доля ТОП-10 доменов: {top_10_links} из {total_links} ({top_10_share:.1f}%)")
    print(f"Энтропия Шеннона: {entropy:.3f} бит (макс. {max_entropy:.3f}, относ. {entropy/max_entropy:.2f})")
    print("\nТоп доменов по частоте цитирования:")
    for d, c in counter.most_common(top_k):
        share = c / total_links * 100
        print(f"  {d:<30} : {c:3d} ({share:4.1f}%)")


def cmd_rewriting(rows: List[Dict[str, Any]], limit: int = 10):
    """Демонстрирует работу модуля переформулирования запросов (Query Rewriting)."""
    ok_rows = [r for r in rows if not r.get("error") and r.get("search_queries")]
    print("=== ДЕКОМПОЗИЦИЯ ИНТЕНТОВ (QUERY REWRITING MODULE) ===")
    print(f"Показано {min(limit, len(ok_rows))} примеров трансформации промпта пользователя:\n")
    for i, r in enumerate(ok_rows[:limit], 1):
        q = r.get("question", "")
        rewritten = r.get("search_queries", [])
        print(f"[{i}] Промпт: \"{q}\"")
        for sub_q in rewritten:
            print(f"    ↳ Поисковый запрос: \"{sub_q}\"")
        print()


def cmd_ask(questions_file: str, out_file: str, runs: int, api_key: str, folder_id: str):
    """Выполняет реальные замеры по списку вопросов."""
    q_path = Path(questions_file)
    if not q_path.exists():
        sys.exit(f"Файл с вопросами не найден: {questions_file}")
        
    questions = [line.strip() for line in q_path.read_text(encoding="utf-8").splitlines() 
                 if line.strip() and not line.startswith("#")]
    total_calls = len(questions) * runs
    est_cost = (total_calls / 1000) * DEFAULT_PRICE_PER_1K
    
    print(f"Загружено вопросов: {len(questions)}")
    print(f"Прогонов на вопрос: {runs} (всего вызовов API: {total_calls})")
    print(f"Оценочная стоимость: ~{est_cost:.2f} ₽")
    
    with open(out_file, "a", encoding="utf-8") as f_out:
        for idx, q in enumerate(questions, 1):
            for r in range(1, runs + 1):
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                print(f"[{idx}/{len(questions)}] Run {r}/{runs}: {q[:50]}...", end="", flush=True)
                res = query_api(q, api_key, folder_id)
                row = {
                    "question": q,
                    "run": r,
                    "ts": ts,
                    **res
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
                print(" ✅" if not res.get("error") else f" ❌ ({res.get('error')})")
                time.sleep(1.1)  # Соблюдение лимита Search API: 1 запрос в секунду


def main():
    parser = argparse.ArgumentParser(description="Yandex Generative Search Probe (RAG Reverse-Engineering Tool)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # 1. Ask
    p_ask = sub.add_parser("ask", help="Выполнить серию регламентных запросов к API")
    p_ask.add_argument("questions", help="Путь к txt-файлу с вопросами (по одному на строку)")
    p_ask.add_argument("--out", required=True, help="Путь для сохранения результата (.jsonl)")
    p_ask.add_argument("--runs", type=int, default=1, help="Количество прогонов на вопрос (default: 1)")
    p_ask.add_argument("--api-key", default=os.getenv("YANDEX_API_KEY"), help="API-Key Yandex Cloud (или env YANDEX_API_KEY)")
    p_ask.add_argument("--folder-id", default=os.getenv("YANDEX_FOLDER_ID"), help="Folder ID Yandex Cloud (или env YANDEX_FOLDER_ID)")

    # 2. Report
    p_rep = sub.add_parser("report", help="Сравнение упоминаемости брендов и цитируемости сайтов")
    p_rep.add_argument("jsonl", help="Путь к JSONL датасету")
    p_rep.add_argument("--brands", required=True, help="Спецификация брендов: 'Имя|домен; Имя2|домен2'")

    # 3. Jaccard
    p_jac = sub.add_parser("jaccard", help="Расчет коэффициента схожести источников Жаккара (стабильность RAG)")
    p_jac.add_argument("jsonl", help="Путь к JSONL датасету с парными прогонами")

    # 4. Entropy
    p_ent = sub.add_parser("entropy", help="Анализ доменной энтропии и концентрации источников")
    p_ent.add_argument("jsonl", help="Путь к JSONL датасету")
    p_ent.add_argument("--top", type=int, default=15, help="Сколько доменов показать в топе")

    # 5. Rewriting
    p_rew = sub.add_parser("rewriting", help="Анализ декомпозиции пользовательских интентов (Query Rewriting)")
    p_rew.add_argument("jsonl", help="Путь к JSONL датасету")
    p_rew.add_argument("--limit", type=int, default=10, help="Количество примеров")

    args = parser.parse_args()

    if args.cmd == "ask":
        if not args.api_key or not args.folder_id:
            sys.exit("Ошибка: Требуются --api-key и --folder-id (или переменные окружения YANDEX_API_KEY / YANDEX_FOLDER_ID)")
        cmd_ask(args.questions, args.out, args.runs, args.api_key, args.folder_id)
    elif args.cmd == "report":
        cmd_report(load_dataset(args.jsonl), args.brands)
    elif args.cmd == "jaccard":
        cmd_jaccard(load_dataset(args.jsonl))
    elif args.cmd == "entropy":
        cmd_entropy(load_dataset(args.jsonl), args.top)
    elif args.cmd == "rewriting":
        cmd_rewriting(load_dataset(args.jsonl), args.limit)


if __name__ == "__main__":
    main()
