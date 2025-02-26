import json
import time
import re
from typing import Dict, List, Tuple, Set, Optional
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import traceback
from dotenv import load_dotenv
load_dotenv()

def generate_targeted_queries(company: str, industry: str) -> Dict[str, List[str]]:
    """
    Generate targeted search queries with a focus on negative events and legal issues,
    organized by category.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        
        system_prompt = (
            "You are an expert forensic investigator specializing in corporate misconduct research. "
            "Create targeted search queries to uncover potential issues for the specified company. "
            "Focus heavily on negative events, especially:\n\n"
            "1. Legal cases AGAINST the company (not cases filed BY the company)\n"
            "2. Fraud or misconduct committed BY the company (not fraud against the company)\n"
            "3. Regulatory investigations and penalties\n"
            "4. Whistleblower allegations\n"
            "5. Accounting irregularities or financial misconduct\n\n"
            
            "Create queries in these specific categories:\n"
            "- 'legal_issues': Queries about lawsuits, litigation against the company\n"
            "- 'regulatory_actions': Queries about investigations, fines, penalties from regulators\n"
            "- 'financial_irregularities': Queries about accounting issues, misstatements, fraud\n"
            "- 'management_misconduct': Queries about executive misbehavior, ethics violations\n"
            "- 'corporate_governance': Queries about board issues, conflicts of interest\n"
            "- 'general': Any other relevant queries not fitting above categories\n\n"
            
            "For each category, provide 2-3 specific search queries using proper search operators.\n"
            "Return a JSON object where keys are categories and values are arrays of query strings.\n"
            "Be precise with language to find events where the company is the accused party, not the victim."
        )
        
        input_message = (
            f"Company: {company}\n"
            f"Industry: {industry}\n\n"
            f"Generate targeted search queries to discover potential misconduct or issues at {company}."
        )
        
        messages = [
            ("system", system_prompt),
            ("human", input_message)
        ]
        
        response = llm.invoke(messages)
        print(f"RAW RESPONSE: {response.content[:500]}")
        response_content = response.content.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
        
        query_categories = json.loads(json_content)
        
        for category, queries in query_categories.items():
            query_categories[category] = [
                q if company.lower() in q.lower() else f'"{company}" {q}' 
                for q in queries
            ]
        
        print(f"[Research Agent] Generated {sum(len(v) for v in query_categories.values())} queries across {len(query_categories)} categories")
        return query_categories
        
    except Exception as e:
        print(f"[Research Agent] Error in query generation: {e}")
        print(traceback.format_exc())
        
        return {
            "legal_issues": [
                f'"{company}" defendant OR sued OR "faces lawsuit" OR litigation',
                f'"{company}" "class action" OR "legal action against"'
            ],
            "regulatory_actions": [
                f'"{company}" investigation OR probe OR "regulatory action"',
                f'"{company}" fine OR penalty OR sanction'
            ],
            "financial_irregularities": [
                f'"{company}" "accounting irregularities" OR fraud OR misstate',
                f'"{company}" "financial misconduct" OR "misleading statements"'
            ],
            "management_misconduct": [
                f'"{company}" executive OR CEO OR management misconduct OR ethics'
            ],
            "corporate_governance": [
                f'"{company}" "board issues" OR "corporate governance" OR "conflict of interest"'
            ],
            "general": [
                f'"{company}" scandal OR controversy OR allegation',
                f'"{company}" "negative news" OR issues OR problem'
            ]
        }

def is_quarterly_report_article(title: str, snippet: str = "") -> bool:
    """
    Determine if an article is about a quarterly or annual financial report.
    """
    title_lower = title.lower()
    snippet_lower = snippet.lower() if snippet else ""
    
    report_terms = [
        'quarterly report', 'q1 report', 'q2 report', 'q3 report', 'q4 report',
        'quarterly results', 'q1 results', 'q2 results', 'q3 results', 'q4 results',
        'quarterly earnings', 'annual report', 'annual results', 'financial results',
        'earnings report', 'quarterly financial', 'year-end results'
    ]
    
    for term in report_terms:
        if term in title_lower or term in snippet_lower:
            return True
    
    if (re.search(r'q[1-4]\s*20[0-9]{2}', title_lower) or 
        re.search(r'fy\s*20[0-9]{2}', title_lower) or
        re.search(r'q[1-4]\s*20[0-9]{2}', snippet_lower) or
        re.search(r'fy\s*20[0-9]{2}', snippet_lower)):
        return True
    
    if (re.search(r'report[s]?\s+\d+%', title_lower) or 
        re.search(r'revenue\s+of\s+[\$£€]', title_lower) or
        re.search(r'profit\s+of\s+[\$£€]', title_lower)):
        return True
    
    return False

def parse_serp_results(raw_output, category: str) -> List[Dict]:
    """
    Enhanced parser that handles both string and list responses from SerpAPI.
    """
    results = []
    
    try:
        if isinstance(raw_output, str):
            print(f"[Research Agent] Attempting to parse string response as JSON...")
            try:
                parsed_data = json.loads(raw_output)
                
                if isinstance(parsed_data, list):
                    raw_output = parsed_data
                    print(f"[Research Agent] Successfully parsed string as JSON list with {len(raw_output)} items")
                elif isinstance(parsed_data, dict) and 'organic_results' in parsed_data:
                    raw_output = parsed_data['organic_results']
                    print(f"[Research Agent] Successfully parsed string as JSON object with organic_results")
            except json.JSONDecodeError:
                print(f"[Research Agent] Not valid JSON, treating as text: {raw_output[:100]}...")
                
                if len(raw_output) > 50:
                    import hashlib
                    hash_id = hashlib.md5(raw_output.encode()).hexdigest()[:10]
                    
                    results.append({
                        "index": 0,
                        "title": raw_output[:100] + "..." if len(raw_output) > 100 else raw_output,
                        "link": f"https://placeholder.com/text_{hash_id}",
                        "date": "Unknown date",
                        "snippet": raw_output,
                        "source": "API text response",
                        "category": category,
                        "is_quarterly_report": False
                    })
                    
                    print(f"[Research Agent] Created result from text response")
                    return results
        
        if isinstance(raw_output, list):
            for i, item in enumerate(raw_output):
                if isinstance(item, dict) and "title" in item and "link" in item:
                    results.append({
                        "index": i,
                        "title": item["title"].strip(),
                        "link": item["link"].strip(),
                        "date": item.get("date", "Unknown date").strip(),
                        "snippet": item.get("snippet", "").strip(),
                        "source": item.get("source", "Unknown source").strip(),
                        "category": category,
                        "is_quarterly_report": False
                    })
        
    except Exception as e:
        print(f"[Research Agent] Error in parse_serp_results: {e}")
        import traceback
        print(traceback.format_exc())
    
    print(f"[Research Agent] Parsed {len(results)} results from category '{category}'")
    return results
def calculate_event_importance(event_name: str, articles: List[Dict]) -> int:
    """
    Calculate an importance score for an event based on content indicators.
    Higher numbers indicate more important events.
    """
    score = 50
    
    event_name_lower = event_name.lower()
    
    if any(term in event_name_lower for term in ['quarterly report', 'financial results', 'earnings report']):
        score -= 30
    
    if any(term in event_name_lower for term in ['fraud', 'lawsuit', 'investigation', 'scandal', 'fine', 'penalty']):
        score += 30
    
    if 'criminal' in event_name_lower:
        score += 40
    
    if 'class action' in event_name_lower:
        score += 25
    
    if any(term in event_name_lower for term in ['regulator', 'sec', 'doj', 'ftc']):
        score += 20
    
    article_count = len(articles)
    score += min(article_count * 5, 25)  
    
    if '- High' in event_name:
        score += 25
    elif '- Medium' in event_name:
        score += 10
    
    reputable_sources = ['bloomberg', 'reuters', 'wsj', 'financial times', 'cnbc', 'nytimes', 'economist']
    for article in articles:
        source = article.get('source', '').lower()
        if any(rep_source in source for rep_source in reputable_sources):
            score += 5
    
    return score

def group_results(company: str, articles: List[Dict], industry: str = None) -> Dict[str, List[Dict]]:
    """
    Group news articles into event clusters with special handling for quarterly reports
    and improved prioritization of negative events.
    """
    if not articles:
        print("[Research Agent] No articles to cluster")
        return {}
    
    quarterly_report_articles = [a for a in articles if a.get("is_quarterly_report", False)]
    other_articles = [a for a in articles if not a.get("is_quarterly_report", False)]
    
    print(f"[Research Agent] Identified {len(quarterly_report_articles)} quarterly report articles")
    print(f"[Research Agent] Processing {len(other_articles)} non-quarterly report articles")
    
    regular_events = {}
    if other_articles:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            
            system_prompt = (
                "You are an expert in analyzing and clustering news articles related to corporate misconduct and issues. "
                "Your task is to group articles into distinct event clusters that represent specific incidents, "
                "investigations, lawsuits, or other forensically-relevant matters.\n\n"
                
                "IMPORTANT GUIDELINES:\n"
                "1. Prioritize negative events where the company is accused of wrongdoing\n"
                "2. Give special attention to lawsuits AGAINST the company (not by the company)\n"
                "3. Focus on fraud committed BY the company (not against the company)\n"
                "4. Distinguish between different lawsuits and regulatory actions\n"
                "5. Create specific event names with dates and severity levels (High/Medium/Low)\n\n"
                
                "EVENT NAMING FORMAT:\n"
                "[Event Type]: [Specific Description] ([Date if available]) - [Severity if known]\n\n"
                
                "Example event names:\n"
                "- 'Fraud Investigation: SEC Probe into Accounting Practices (March 2023) - High'\n"
                "- 'Class Action Lawsuit: Misleading Statements to Investors (Q2 2022) - Medium'\n"
                "- 'Regulatory Fine: EPA Penalty for Environmental Violations (2021) - Medium'\n\n"
                
                "Return a JSON object where:\n"
                "- Each key is a precise event name following the format above\n"
                "- Each value is a list of INTEGER indices from the article list\n"
            )

            simplified_articles = []
            for i, article in enumerate(other_articles):
                simplified_articles.append({
                    "index": i,
                    "title": article["title"],
                    "snippet": article.get("snippet", ""),
                    "date": article.get("date", "Unknown date"),
                    "source": article.get("source", "Unknown source"),
                    "category": article.get("category", "general")
                })

            input_message = (
                f"Company: {company}\n"
                f"Industry: {industry if industry else 'Unknown'}\n"
                f"Articles to analyze: {json.dumps(simplified_articles)}\n\n"
                f"Group these articles about {company} into distinct event clusters, focusing on negative events, lawsuits against the company, and potential misconduct."
            )

            messages = [
                ("system", system_prompt),
                ("human", input_message)
            ]

            response = llm.invoke(messages)
            response_content = response.content.strip()
            
            if "```json" in response_content:
                json_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_content = response_content.split("```")[1].strip()
            else:
                json_content = response_content
                
            clustered_indices = json.loads(json_content)
            
            for event_name, indices in clustered_indices.items():
                valid_indices = []
                for idx in indices:
                    if isinstance(idx, str) and idx.isdigit():
                        idx = int(idx)
                    if isinstance(idx, int) and 0 <= idx < len(other_articles):
                        valid_indices.append(idx)
                
                if valid_indices:
                    regular_events[event_name] = [other_articles[i] for i in valid_indices]
            
            print(f"[Research Agent] Grouped non-quarterly articles into {len(regular_events)} events")
            
        except Exception as e:
            print(f"[Research Agent] Error clustering non-quarterly articles: {e}")
            print(traceback.format_exc())
            
            for i, article in enumerate(other_articles):
                event_name = f"News: {article['title'][:50]}..."
                regular_events[event_name] = [article]
    
    if quarterly_report_articles:
        dates = [article.get("date", "") for article in quarterly_report_articles]
        valid_dates = [d for d in dates if d and d.lower() != "unknown date"]
        
        date_str = ""
        if valid_dates:
            try:
                parsed_dates = []
                for date_text in valid_dates:
                    try:
                        for fmt in ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]:
                            try:
                                parsed_date = datetime.strptime(date_text, fmt)
                                parsed_dates.append(parsed_date)
                                break
                            except:
                                continue
                    except:
                        pass
                
                if parsed_dates:
                    most_recent = max(parsed_dates)
                    date_str = f" ({most_recent.strftime('%b %Y')})"
            except:
                date_str = f" ({valid_dates[0]})"
        
        quarterly_event_name = f"Financial Reporting: Quarterly/Annual Results{date_str} - Low"
        regular_events[quarterly_event_name] = quarterly_report_articles
        print(f"[Research Agent] Created a consolidated event for {len(quarterly_report_articles)} quarterly report articles")
    
    final_events = {}
    importance_scores = {}
    
    for event_name, event_articles in regular_events.items():
        importance = calculate_event_importance(event_name, event_articles)
        importance_scores[event_name] = importance
        
        event_data = {
            "articles": event_articles,
            "importance_score": importance,
            "article_count": len(event_articles)
        }
        final_events[event_name] = event_data
    
    print(f"[Research Agent] Assigned importance scores to {len(importance_scores)} events")
    for event, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"[Research Agent] Event: '{event}' - Score: {score}")
    
    return final_events

def research_agent(state: Dict) -> Dict:
    """
    Enhanced research agent with improved error handling, especially for SerpAPI responses.
    """
    print("[Research Agent] Received state:", state)
    company = state.get("company", "")
    industry = state.get("industry", "Unknown")
    
    if not company:
        print("[Research Agent] ERROR: Missing company name")
        return {**state, "goto": "meta_agent", "error": "Missing company name"}
    
    print(f"[Research Agent] Beginning targeted research for {company} (Industry: {industry})")
    
    query_categories = generate_targeted_queries(company, industry)
    
    all_articles = []
    
    for category, queries in query_categories.items():
        print(f"[Research Agent] Processing category: {category}")
        
        for query in queries:
            print(f"[Research Agent] Executing search query: {query}")
            params = {
                "engine": "google",
                "q": query,
                "location": "India",
                "google_domain": "google.co.in",
                "gl": "in",
                "hl": "en",
                "safe": "off", 
                "num": "100",
                "tbm": "nws",
                "output": "json"  
            }
            
            try:
                serp = SerpAPIWrapper(params=params)
                raw_output = serp.run(query)

                print(f"[DIAGNOSTIC] Raw output type immediately after serp.run(): {type(raw_output)}")
                if isinstance(raw_output, list) and len(raw_output) > 0:
                    print(f"[DIAGNOSTIC] First item keys: {list(raw_output[0].keys()) if isinstance(raw_output[0], dict) else 'Not a dict'}")
                
                print(f"[Research Agent] SerpAPI returned data of type: {type(raw_output)}")
                if isinstance(raw_output, str) and len(raw_output) > 200:
                    print(f"[Research Agent] First 200 chars of response: {raw_output[:200]}")
                elif isinstance(raw_output, dict):
                    print(f"[Research Agent] Response keys: {list(raw_output.keys())}")
                elif isinstance(raw_output, list):
                    print(f"[Research Agent] Response is a list with {len(raw_output)} items")
                
                articles = parse_serp_results(raw_output, category)
                all_articles.extend(articles)
                
                time.sleep(1)
            except Exception as e:
                print(f"[Research Agent] Error executing query '{query}': {e}")
                import traceback
                print(traceback.format_exc())
    
    print(f"[Research Agent] Collected {len(all_articles)} total articles across all categories")
    

    if not all_articles:
        print("[Research Agent] No articles found with targeted queries. Trying fallback query.")
        try:
            fallback_query = f'"{company}" news'
            params = {
                "engine": "google",
                "q": fallback_query,
                "num": "100",
                "location": "India",
                "google_domain": "google.co.in",
                "gl": "in"
            }
            
            serp = SerpAPIWrapper(params=params)
            raw_output = serp.run(fallback_query)
            fallback_articles = parse_serp_results(raw_output, "general")
            all_articles.extend(fallback_articles)
            print(f"[Research Agent] Fallback query returned {len(fallback_articles)} articles")
        except Exception as e:
            print(f"[Research Agent] Error with fallback query: {e}")
    
    unique_articles = []
    seen_urls = set()
    
    for article in all_articles:
        if article["link"] not in seen_urls:
            seen_urls.add(article["link"])
            unique_articles.append(article)
    
    print(f"[Research Agent] Deduplicated to {len(unique_articles)} unique articles")
    
    if not unique_articles:
        print("[Research Agent] ERROR: No articles found after all attempts")
        return {**state, "goto": "meta_agent", "error": "No articles found"}
    
    if len(unique_articles) <= 3:
        print("[Research Agent] Small number of articles, using simplified grouping")
        grouped_results = {}
        for i, article in enumerate(unique_articles):
            event_name = f"News: {article['title'][:50]}..."
            grouped_results[event_name] = {
                "articles": [article],
                "importance_score": 50,
                "article_count": 1
            }
    else:
        grouped_results = group_results(company, unique_articles, industry)
    
    sorted_events = sorted(
        grouped_results.items(), 
        key=lambda x: x[1]["importance_score"], 
        reverse=True
    )
    
    print(f"[Research Agent] Identified and ranked {len(grouped_results)} distinct events")
    
    final_results = {}
    for event_name, event_data in sorted_events:
        final_results[event_name] = event_data["articles"]
    
    event_metadata = {
        event_name: {
            "importance_score": event_data["importance_score"],
            "article_count": event_data["article_count"],
            "is_quarterly_report": any(a.get("is_quarterly_report", False) for a in event_data["articles"])
        }
        for event_name, event_data in grouped_results.items()
    }
    
    state["research_results"] = final_results
    state["event_metadata"] = event_metadata
    print("[Research Agent] Updated state with research results and event metadata")
    
    return {**state, "goto": "meta_agent"}