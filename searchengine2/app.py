
# if user_query:
#     if st.session_state.agent is None:
#         st.warning("Please enter your Groq API key first.")
#     else:
#         response = st.session_state.agent.run(user_query)

#         st.session_state.chat_history.append((user_query, response))

#         st.markdown("### 🧾 Response")
#         st.write(response)

# # -----------------------------
# # CHAT HISTORY DISPLAY
# # -----------------------------
# if st.session_state.chat_history:
#     st.markdown("---")
#     st.subheader("💬 Conversation History")

#     for q, r in reversed(st.session_state.chat_history):
#         st.markdown(f"**You:** {q}")
#         st.markdown(f"**AI:** {r}")
#         st.markdown("---")