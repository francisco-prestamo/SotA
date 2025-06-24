import streamlit as st
import backend_api

st.set_page_config(page_title="SotA - Chat", layout="centered")
st.title("SotA")

if "state" not in st.session_state:
    st.session_state.state = "start"
    st.session_state.thesis_input = ""
    st.session_state.answers = {}
    st.session_state.current_questions = []
    st.session_state.expert_question = None

def render_chat():
    for speaker, msg in backend_api.get_chat_history():
        if speaker == "User":
            with st.chat_message("user"):
                st.markdown(msg)
        else:
            with st.chat_message(speaker):
                st.markdown(msg)

render_chat()

if st.session_state.state == "start":
    initial_msg = backend_api.get_initial_prompt()
    st.session_state.state = "waiting_topic"
    st.rerun()

elif st.session_state.state == "waiting_topic":
    with st.chat_message("user"):
        topic = st.text_input("Enter your thesis topic or article description", key="topic_input")
        if st.button("Submit Topic"):
            if topic.strip():
                st.session_state.thesis_input = topic
                backend_api.conversation_history.append(("User", topic.strip()))
                st.session_state.state = "clarify"
                st.rerun()

elif st.session_state.state == "clarify":
    questions = backend_api.get_follow_up_questions(st.session_state.thesis_input)
    st.session_state.current_questions = questions
    st.session_state.answers = {}
    st.session_state.state = "answer_questions"
    st.rerun()

elif st.session_state.state == "answer_questions":
    with st.chat_message("user"):
        with st.form("clarify_form"):
            for q in st.session_state.current_questions:
                st.session_state.answers[q] = st.text_input(q, key=q)
            submitted = st.form_submit_button("Submit Answers")
            if submitted:
                backend_api.submit_clarification_answers(st.session_state.answers)
                st.session_state.state = "show_table"
                st.rerun()

elif st.session_state.state == "show_table":
    table_md = backend_api.get_state_of_the_art_table()
    st.session_state.state = "expert_question"
    st.rerun()

elif st.session_state.state == "expert_question":
    question = backend_api.ask_for_clarification_about_table()
    st.session_state.expert_question = question
    st.session_state.state = "await_user_reply"
    st.rerun()

elif st.session_state.state == "await_user_reply":
    with st.chat_message("user"):
        answer = st.radio(st.session_state.expert_question, ["Yes", "No", "Not sure"], key="expert_answer")
        if st.button("Submit Answer"):
            backend_api.user_response_to_expert(st.session_state.expert_question, answer)
            st.success("Response sent to experts.")
            st.session_state.state = "done"

elif st.session_state.state == "done":
    with st.chat_message("Recepcionist"):
        st.markdown("✅ The expert panel has finalized the table. Thank you!")
    if st.button("🔁 Start Over"):
        st.session_state.clear()
        st.rerun()
