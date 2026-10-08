import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

st.set_page_config(page_title="NDA Clause Segmentation Review", layout="wide")

CSV_PATH = r"f:\NDA\data\human_review\human_review_required.csv"
STATS_PATH = r"f:\NDA\reports\human_review\review_statistics.csv"

@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        st.error(f"File not found: {CSV_PATH}")
        return pd.DataFrame()

def save_data(df):
    df.to_csv(CSV_PATH, index=False)

def update_statistics(review_method):
    if not os.path.exists(STATS_PATH):
        return
        
    stats_df = pd.read_csv(STATS_PATH)
    if stats_df.empty:
        return
        
    if "ai_suggestions_generated" not in stats_df.columns:
        stats_df["ai_suggestions_generated"] = 0
        stats_df["ai_accepted"] = 0
        stats_df["ai_overridden"] = 0
        stats_df["manual_without_ai"] = 0
        
    stats_df.at[0, "reviewed_segments"] = stats_df.at[0, "reviewed_segments"] + 1
    stats_df.at[0, "unreviewed_segments"] = max(0, stats_df.at[0, "total_segments"] - stats_df.at[0, "reviewed_segments"])

    if review_method == "human_accepted_ai":
        stats_df.at[0, "ai_accepted"] = stats_df.at[0, "ai_accepted"] + 1
    elif review_method == "human_override_ai":
        stats_df.at[0, "ai_overridden"] = stats_df.at[0, "ai_overridden"] + 1
    elif review_method == "manual_without_ai":
        stats_df.at[0, "manual_without_ai"] = stats_df.at[0, "manual_without_ai"] + 1
        
    stats_df.to_csv(STATS_PATH, index=False)


def main():
    st.title("NDA Clause Segmentation - Human Audit")
    
    df = load_data()
    if df.empty:
        return

    # Ensure necessary columns exist in the DataFrame
    expected_cols = [
        "human_is_valid_clause", "human_corrected_text", 
        "human_action", "human_notes", "reviewer", "reviewed",
        "ai_suggested_action", "ai_confidence", "ai_reason",
        "review_method", "review_timestamp", "previous_text", "next_text", "review_reason"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    # Force string columns to object dtype
    string_cols = [
        "human_action", "human_corrected_text", "human_notes", "reviewer", 
        "human_is_valid_clause", "ai_suggested_action", "ai_reason", 
        "review_method", "review_timestamp", "ai_confidence", "previous_text", "next_text", "review_reason"
    ]
    for col in string_cols:
        df[col] = df[col].astype('object')

    # Ensure reviewed is boolean type
    if df["reviewed"].dtype != bool:
        df["reviewed"] = df["reviewed"].astype(str).str.lower() == "true"

    total_segments = len(df)
    reviewed_count = df["reviewed"].sum()

    st.sidebar.header("Progress")
    st.sidebar.text(f"Reviewed: {reviewed_count} / {total_segments}")
    st.sidebar.progress(reviewed_count / total_segments if total_segments > 0 else 0)

    if 'current_index' not in st.session_state:
        # Find first unreviewed index
        unreviewed = df.index[~df['reviewed']].tolist()
        st.session_state.current_index = unreviewed[0] if unreviewed else 0
        st.session_state.change_decision = False

    idx = st.session_state.current_index
    
    if idx >= total_segments:
        st.success("All segments in the audit queue have been reviewed!")
        if st.button("Start Over"):
            st.session_state.current_index = 0
            st.session_state.change_decision = False
            st.rerun()
        return

    row = df.iloc[idx]
    
    st.subheader(f"Audit Segment {idx + 1} of {total_segments}")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### SEGMENT INFORMATION")
        st.markdown("---")
        st.markdown(f"**Document ID:** `{row['document_id']}`")
        st.markdown(f"**Clause ID:** `{row['clause_id']}`")
        st.markdown(f"**Segment Type:** `{row['segment_type']}`")
        st.markdown(f"**Length:** `{row['length']}` chars")
        st.markdown(f"**Review Reason:** `{row['review_reason']}`")
        
        st.markdown("### Context")
        st.text_area("Previous Segment Text", value=row.get("previous_text", "N/A"), height=150, disabled=True)
        st.text_area("Current Clause Text", value=row["clause_text"], height=300, disabled=True)
        st.text_area("Next Segment Text", value=row.get("next_text", "N/A"), height=150, disabled=True)

    with col2:
        st.markdown("### AI ASSISTANCE")
        st.markdown("---")
        
        st.info(f"**Suggested Action:** {row.get('ai_suggested_action', 'N/A')}")
        conf = row.get('ai_confidence')
        if pd.notna(conf) and conf != "":
            try:
                st.info(f"**Confidence:** {float(conf) * 100:.1f}%")
            except:
                st.info(f"**Confidence:** {conf}")
        st.info(f"**Reason:** {row.get('ai_reason', 'N/A')}")
        
        col_acc, col_chg = st.columns(2)
        with col_acc:
            if st.button("ACCEPT AI SUGGESTION", type="primary"):
                act = row.get('ai_suggested_action')
                if pd.isna(act) or act == "":
                    act = "REVIEW"
                    
                df.at[idx, "human_action"] = act
                if act in ["KEEP", "MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS", "SPLIT"]:
                    df.at[idx, "human_is_valid_clause"] = True
                elif act == "REMOVE_NON_LEGAL":
                    df.at[idx, "human_is_valid_clause"] = False
                else:
                    df.at[idx, "human_is_valid_clause"] = ""
                    
                df.at[idx, "human_corrected_text"] = ""
                df.at[idx, "human_notes"] = row.get('ai_reason', '')
                df.at[idx, "reviewer"] = "Human" 
                df.at[idx, "reviewed"] = True
                
                df.at[idx, "review_method"] = "human_accepted_ai"
                df.at[idx, "review_timestamp"] = datetime.now().isoformat()
                
                save_data(df)
                update_statistics("human_accepted_ai")
                
                st.session_state.current_index += 1
                st.session_state.change_decision = False
                load_data.clear()
                st.rerun()
                
        with col_chg:
            if st.button("CHANGE DECISION"):
                st.session_state.change_decision = True

        if st.session_state.get("change_decision"):
            st.markdown("### HUMAN FINAL DECISION")
            st.markdown("---")
            
            action_options = ["KEEP", "MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS", "SPLIT", "REMOVE_NON_LEGAL", "REVIEW"]
            current_action = row["human_action"] if row["human_action"] in action_options else 0
            if current_action == 0 and pd.notna(row.get('ai_suggested_action')) and row.get('ai_suggested_action') in action_options:
                current_action = action_options.index(row.get('ai_suggested_action'))
            elif current_action in action_options:
                current_action = action_options.index(current_action)
            else:
                current_action = 0
                
            action = st.radio(
                "Action",
                options=action_options,
                index=current_action,
                key=f"action_{idx}"
            )
            
            corrected_text = st.text_area(
                "Human Corrected Text (Optional)", 
                value=row["human_corrected_text"] if pd.notna(row["human_corrected_text"]) else "", 
                key=f"corr_{idx}"
            )
            notes = st.text_area(
                "Notes (Optional)", 
                value=row["human_notes"] if pd.notna(row["human_notes"]) else "", 
                key=f"notes_{idx}"
            )
            reviewer = st.text_input(
                "Reviewer Name", 
                value=row["reviewer"] if pd.notna(row["reviewer"]) else "Human", 
                key=f"rev_{idx}"
            )

            st.markdown("---")
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("Previous Segment", disabled=idx == 0):
                    st.session_state.current_index -= 1
                    st.session_state.change_decision = False
                    st.rerun()
                    
            with col_next:
                if st.button("Save & Next", type="primary"):
                    if action in ["KEEP", "MERGE_WITH_NEXT", "MERGE_WITH_PREVIOUS", "SPLIT"]:
                        is_valid = True
                    elif action == "REMOVE_NON_LEGAL":
                        is_valid = False
                    else: # REVIEW
                        is_valid = ""

                    df.at[idx, "human_action"] = action
                    df.at[idx, "human_is_valid_clause"] = is_valid
                    df.at[idx, "human_corrected_text"] = corrected_text
                    df.at[idx, "human_notes"] = notes
                    df.at[idx, "reviewer"] = reviewer
                    df.at[idx, "reviewed"] = True
                    df.at[idx, "review_timestamp"] = datetime.now().isoformat()
                    
                    if pd.notna(row.get('ai_suggested_action')) and row.get('ai_suggested_action') != "":
                        df.at[idx, "review_method"] = "human_override_ai"
                        review_method = "human_override_ai"
                    else:
                        df.at[idx, "review_method"] = "manual_without_ai"
                        review_method = "manual_without_ai"
                    
                    save_data(df)
                    update_statistics(review_method)
                    
                    st.session_state.current_index += 1
                    st.session_state.change_decision = False
                    load_data.clear()
                    st.rerun()

if __name__ == "__main__":
    main()
