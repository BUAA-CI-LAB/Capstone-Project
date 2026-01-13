import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
import re

BASE = "https://pubs.usgs.gov"
START_URL = BASE + "/browse/Report/USGS%20Numbered%20Series/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

SLEEP = 0.8  # 放慢点，稳


# ---------- 工具函数 ----------

def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    return BeautifulSoup(r.text, "lxml")


def clean_text(s):
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------- Step 1：获取所有 Report 类别 ----------

def get_all_categories(w):
    """
    从 USGS Numbered Series 页面提取所有类别
    """
    soup = get_soup(w)
    if not soup:
        raise RuntimeError("Cannot access start page")

    categories = {}

    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue

        if "/browse/Report/USGS%20Numbered%20Series/" in href:
            name = clean_text(a.get_text())
            url = urljoin(BASE, href)

            # 排除返回上级的链接
            if name and name.lower() not in ("up", "parent directory"):
                categories[name] = url

    return categories


# ---------- Step 2：获取某个类别下的所有子页面（年份 / 分页） ----------

def get_subpages(category_url):
    """
    返回该类别下所有子页面（年份目录等）
    """
    soup = get_soup(category_url)
    if not soup:
        return []

    pages = set([category_url])
    print(f"page: {pages}")
    for a in soup.select("a[href]"):
        href = a.get("href")
        if href and href.startswith(category_url):
            pages.add(href)

    return sorted(pages)


# ---------- Step 3：从列表页提取 report publication 链接 ----------

def get_publication_links(list_url):
    """
    从某个列表页中，找出所有 publication 页面链接
    """
    soup = get_soup(list_url)
    if not soup:
        return []

    pubs = set()

    for a in soup.select("a[href]"):
        href = a.get("href")
        if href and "/publication/" in href:
            pubs.add(urljoin(BASE, href))

    return sorted(pubs)


# ---------- Step 4：从 publication 页面提取摘要 ----------

def extract_abstract(pub_url):
    soup = get_soup(pub_url)
    if not soup:
        return None

    # 1️⃣ 找 Abstract 标题（h4 为主，兼容 h3）
    abstract_h = None
    for h in soup.find_all(["h4", "h3"]):
        if h.get_text(strip=True).lower() == "abstract":
            abstract_h = h
            break

    if not abstract_h:
        return None

    # 2️⃣ 从 Abstract 开始，顺序遍历 DOM，直到下一个 h4/h3
    texts = []
    for el in abstract_h.next_elements:
        # 遇到下一个同级标题，停止
        if el.name in ["h4", "h3"] and el is not abstract_h:
            break

        if el.name == "p":
            txt = clean_text(el.get_text())
            if txt and len(txt) > 50:
                texts.append(txt)

    if not texts:
        return None

    return " ".join(texts)

def is_usgs_noise_abstract(text: str) -> bool:
    """
    判断是否为 USGS Publications Warehouse 的模板/噪声摘要
    """
    t = text.lower()

    noise_patterns = [
        "portable document format",
        "presented in portable document format",
        "available in pdf format",
        "pdf format (pdf)",
        "click here to download",
        "download the pdf",
        "this report is available",
        "part or all of this report",
    ]

    return any(p in t for p in noise_patterns)

def debug_headers(pub_url):
    soup = get_soup(pub_url)
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        print("TAG:", h.name, "| TEXT:", repr(h.get_text(strip=True)))


# ---------- 主流程 ----------

def main():
    categories = get_all_categories(START_URL)
    print(f"Found {len(categories)} categories")

    results = []
    seen_publications = set()
    stop = False
    start = False
    for cat_name, cat_url in categories.items():
        print(f"\n[Category] {cat_name}")
        if cat_name == 'Water Data Report':
            start = True
            continue
            

        if start == False:
            continue
        subpages = get_subpages(cat_url)

        for page in subpages:
            print(f"  Scanning: {page}")
            r = get_all_categories(page)
            for p_name, p in r.items():
                print(p)
                pub_links = get_publication_links(p)
                print(len(pub_links))
                for pub in pub_links:
                    if pub in seen_publications:
                        continue
                    seen_publications.add(pub)

                    abstract = extract_abstract(pub)

                    if abstract and not is_usgs_noise_abstract(abstract):
                        record = {
                            "category": cat_name,
                            "publication_url": pub,
                            "abstract": abstract
                        }
                        results.append(record)
                        print(record)
                        print(f"get {len(results)}/{10000}...", end="\r")
                        if len(results) > 10000:
                            stop = True
                            break
                        if len(results) % 20 == 0:
                            with open("usgs_reports_abstracts_6.jsonl", "w", encoding="utf-8") as f:
                                for r in results:
                                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    time.sleep(SLEEP)
                    if stop:
                        break
        if stop:
            break       
            
        
    with open("usgs_reports_abstracts_6.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Total reports processed: {len(results)}")
    print("Saved to usgs_reports_abstracts.jsonl")


if __name__ == "__main__":
    main()
