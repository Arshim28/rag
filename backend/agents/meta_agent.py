import json
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

def evaluate_research_quality(company: str, research_results: Dict) -> Dict:
    """
    Evaluate the quality, completeness, and balance of the research results.
    Returns a quality assessment with scores and recommendations.
    """
    if not research_results:
        return {
            "overall_score": 0,
            "coverage_score": 0,
            "balance_score": 0,
            "recency_score": 0,
            "credibility_score": 0,
            "assessment": "No research results available.",
            "recommendations": ["Initiate research to gather information on the company."]
        }
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert research quality assessor. Evaluate the research results for completeness, "
        "balance, recency, and credibility. Consider these factors:\n\n"
        "1. COVERAGE: Are major areas covered (operational, legal, management, reputation)?\n"
        "2. BALANCE: Is there a mix of positive, neutral, and concerning information?\n"
        "3. RECENCY: Are the events recent enough to be relevant?\n"
        "4. CREDIBILITY: Are the sources reliable and substantial?\n\n"
        "Return a JSON object with:\n"
        "- 'overall_score': 0-10 rating of overall research quality\n"
        "- 'coverage_score': 0-10 rating of topic coverage\n"
        "- 'balance_score': 0-10 rating of information balance\n"
        "- 'recency_score': 0-10 rating of information timeliness\n"
        "- 'credibility_score': 0-10 rating of source reliability\n"
        "- 'assessment': Brief explanation of the assessment\n"
        "- 'recommendations': List of specific research areas needing improvement"
    )
    
    input_message = (
        f"Company: {company}\n"
        f"Research Results: {json.dumps(research_results)}\n\n"
        f"Evaluate the quality of these research results for {company}."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        assessment = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return assessment
    except Exception as e:
        print(f"[Meta Agent] Error in research quality evaluation: {e}")
        return {
            "overall_score": 5,  
            "coverage_score": 5,
            "balance_score": 5,
            "recency_score": 5,
            "credibility_score": 5,
            "assessment": "Unable to evaluate research quality due to an error.",
            "recommendations": ["Continue with available research while addressing technical issues."]
        }

def identify_research_gaps(company: str, industry: str, research_results: Dict) -> List[str]:
    """
    Identify specific gaps in the research that should be addressed.
    Returns a list of specific research directions.
    """
    if not research_results:
        return ["General company background", "Recent financial performance", "Management team", 
                "Business model", "Competitive position", "Regulatory environment"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert corporate intelligence analyst. Based on the existing research results, "
        "identify specific gaps that should be addressed for a comprehensive analysis of the company. "
        "Consider these potential areas:\n\n"
        "1. Regulatory compliance record\n"
        "2. Legal issues and litigation history\n"
        "3. Corporate governance practices\n"
        "4. Competitive position in the industry\n"
        "5. Reputation and customer satisfaction\n"
        "6. Environmental, social, and governance (ESG) factors\n"
        "7. Innovation and R&D pipeline\n"
        "8. International operations and risks\n\n"
        "Return a JSON array of specific research queries that would address the most critical gaps."
    )
    
    event_types = list(research_results.keys())
    
    input_message = (
        f"Company: {company}\n"
        f"Industry: {industry}\n"
        f"Current Research Coverage: {json.dumps(event_types)}\n\n"
        f"Identify the most important information gaps for {company} given the current research."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        gaps = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return gaps
    except Exception as e:
        print(f"[Meta Agent] Error in identifying research gaps: {e}")
        return ["General company information", "Recent financial performance", "Management team", 
                "Business model", "Competitive position", "Regulatory environment"]

def create_research_plan(company: str, gaps: List[str]) -> Dict[str, List[str]]:
    """
    Create a structured research plan based on identified gaps.
    Returns a dictionary with research categories and specific queries.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert research strategist specializing in corporate intelligence. "
        "Based on the identified information gaps, create a structured research plan with "
        "specific search queries organized into categories. For each gap, generate 2-3 "
        "targeted search queries that would yield high-quality information.\n\n"
        "Return a JSON object where:\n"
        "- Keys are research categories (e.g., 'Financial Performance', 'Legal Issues')\n"
        "- Values are arrays of specific search queries formatted for search engines"
    )
    
    input_message = (
        f"Company: {company}\n"
        f"Identified Information Gaps: {json.dumps(gaps)}\n\n"
        f"Create a structured research plan to address these gaps for {company}."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        plan = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return plan
    except Exception as e:
        print(f"[Meta Agent] Error in creating research plan: {e}")
        return {
            "Company Background": [f'"{company}" history', f'"{company}" founding', f'"{company}" overview'],
            "Financial Performance": [f'"{company}" revenue', f'"{company}" financial results', f'"{company}" earnings'],
            "Management Team": [f'"{company}" CEO', f'"{company}" executives', f'"{company}" leadership team'],
            "Legal Issues": [f'"{company}" lawsuit', f'"{company}" legal', f'"{company}" litigation']
        }

def generate_analysis_guidance(company: str, research_results: Dict) -> Dict:
    """
    Generate guidance for the analyst agent based on research results.
    Returns structured guidance with focus areas and analysis strategies.
    """
    if not research_results:
        return {
            "focus_areas": ["General company assessment"],
            "priorities": ["Establish baseline understanding of company"],
            "analysis_strategies": ["Conduct general background research"],
            "red_flags": ["Insufficient information available"]
        }
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert in corporate forensics and due diligence. Based on the research results, "
        "provide structured guidance for the analysis phase. Identify the most important areas to "
        "focus on, potential red flags that require deeper investigation, and effective analytical "
        "strategies for this specific company.\n\n"
        "Return a JSON object with:\n"
        "- 'focus_areas': List of specific topics that deserve primary attention\n"
        "- 'priorities': List of events/issues ranked by importance for analysis\n"
        "- 'analysis_strategies': List of specific approaches to extract maximum insights\n"
        "- 'red_flags': List of concerning patterns or issues requiring deeper investigation\n"
        "- 'context_recommendations': List of additional context needed for proper analysis"
    )
    
    input_message = (
        f"Company: {company}\n"
        f"Research Results: {json.dumps(research_results)}\n\n"
        f"Generate analysis guidance for {company} based on these research results."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        guidance = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return guidance
    except Exception as e:
        print(f"[Meta Agent] Error in generating analysis guidance: {e}")
        return {
            "focus_areas": ["Overview of all identified events"],
            "priorities": ["Most recent events", "Events with highest impact ratings"],
            "analysis_strategies": ["Compare events chronologically", "Look for patterns in company behavior"],
            "red_flags": ["Any recurring issues", "Any regulatory actions"],
            "context_recommendations": ["Industry context", "Company history"]
        }

def meta_agent(state: Dict) -> Dict:
    """
    Intelligent orchestration agent that evaluates research quality,
    identifies gaps, creates research plans, and provides guidance.
    """
    print("[Meta Agent] Received state:", state)
    
    company = state.get("company", "")
    if not company:
        print("[Meta Agent] ERROR: 'company' key is missing!")
        return {**state, "goto": "END", "error": "Company name is missing"}
    
    research_results = state.get("research_results", {})
    industry = state.get("industry", "Unknown")
    
    print("[Meta Agent] Evaluating research quality...")
    quality_assessment = evaluate_research_quality(company, research_results)
    print(f"[Meta Agent] Research quality score: {quality_assessment.get('overall_score', 0)}/10")
    print(f"[Meta Agent] Assessment: {quality_assessment.get('assessment', 'N/A')}")
    
    if len(research_results) < 3 or quality_assessment.get('overall_score', 0) < 6:
        print("[Meta Agent] Research quality insufficient. Identifying gaps...")
        
        gaps = identify_research_gaps(company, industry, research_results)
        print(f"[Meta Agent] Identified {len(gaps)} research gaps: {gaps[:3]}...")
        
        research_plan = create_research_plan(company, gaps)
        print(f"[Meta Agent] Created research plan with {len(research_plan)} categories")
        
        state["research_plan"] = research_plan
        state["quality_assessment"] = quality_assessment
        
        print("[Meta Agent] Routing to Research Agent with targeted research plan")
        return {**state, "goto": "research_agent"}
    else:
        print("[Meta Agent] Research quality sufficient. Generating analysis guidance...")
        
        analysis_guidance = generate_analysis_guidance(company, research_results)
        print(f"[Meta Agent] Generated analysis guidance with {len(analysis_guidance.get('focus_areas', []))} focus areas")
        
        state["analysis_guidance"] = analysis_guidance
        state["quality_assessment"] = quality_assessment
        
        print("[Meta Agent] Routing to Analyst Agent with analysis guidance")
        return {**state, "goto": "analyst_agent"}