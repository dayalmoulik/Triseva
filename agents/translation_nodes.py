from utils.translation_helper import is_hindi_or_hinglish, translate
import re
from agents.state import TriSevaState

def translation_pre_node(state: TriSevaState) -> TriSevaState:
    """Pre-processing node: Translates user query (and image text if present) 
    to English if they are Hindi or Hinglish."""
    query = state.get("user_query", "")
    
    # Detect the script used (Devanagari vs Latin)
    is_devanagari = bool(re.search(r"[\u0900-\u097f]", query))
    script_hint = "devanagari" if is_devanagari else "latin"
    state["script_hint"] = script_hint
    
    # Check if query is in Hindi/Hinglish
    if is_hindi_or_hinglish(query):
        print(f"  [Translation Pre] Hindi/Hinglish query detected: '{query}' ({script_hint} script)")
        translated_query = translate(query, source_lang="hi-IN", target_lang="en-IN")
        print(f"  [Translation Pre] Translated to English: '{translated_query}'")
        
        # Save original details in state for post-processing
        state["original_language"] = "hi-IN"
        state["original_query"] = query
        state["user_query"] = translated_query
    else:
        state["original_language"] = "en-IN"
        state["original_query"] = query

    # Also check and translate extracted image text if present and in Hindi
    image_text = state.get("image_text")
    if image_text and is_hindi_or_hinglish(image_text):
        print(f"  [Translation Pre] Hindi text in uploaded image detected. Translating to English...")
        translated_img = translate(image_text, source_lang="hi-IN", target_lang="en-IN")
        state["image_text"] = translated_img

    return state

def translation_post_node(state: TriSevaState) -> TriSevaState:
    """Post-processing node: Translates final answers (and draft answers) 
    back to Hinglish/Hindi if the original query was Hindi or Hinglish."""
    original_lang = state.get("original_language", "en-IN")
    script_hint = state.get("script_hint", "latin")
    
    if original_lang == "hi-IN":
        final_ans = state.get("final_answer")
        draft_ans = state.get("draft_answer")
        
        if final_ans:
            print(f"[Translation Post] Translating final answer back to Hinglish ({script_hint} script)...")
            translated_final = translate(final_ans, source_lang="en-IN", target_lang="hi-IN", script_hint=script_hint)
            state["final_answer"] = translated_final
            
        if draft_ans:
            translated_draft = translate(draft_ans, source_lang="en-IN", target_lang="hi-IN", script_hint=script_hint)
            state["draft_answer"] = translated_draft
            
        # Also translate quiz questions if generated
        quiz = state.get("quiz")
        if quiz:
            print(f"[Translation Post] Translating quiz questions back to Hinglish ({script_hint} script)...")
            translated_quiz = []
            for item in quiz:
                t_q = translate(item.get("question", ""), source_lang="en-IN", target_lang="hi-IN", script_hint=script_hint)
                t_opts = [translate(opt, source_lang="en-IN", target_lang="hi-IN", script_hint=script_hint) for opt in item.get("options", [])]
                t_ans = translate(item.get("answer", ""), source_lang="en-IN", target_lang="hi-IN", script_hint=script_hint)
                translated_quiz.append({
                    "question": t_q,
                    "options": t_opts,
                    "answer": t_ans
                })
            state["quiz"] = translated_quiz
            
    return state
