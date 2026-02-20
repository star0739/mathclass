    # -------------------------
    # 하단(전체 폭): ③ 서술 + 백업 + 저장 + 저장상태
    # -------------------------
    st.divider()
    st.subheader("③ 관찰 기록 서술")

    st.markdown(
        r"""
1) 손실함수 $E(a,b)=\alpha a^2+\beta b^2$에 대해 $\dfrac{\partial E}{\partial a}$, $\dfrac{\partial E}{\partial b}$를 구하시오.  
예: 각 변수에 대해 미분하여 얻은 식을 간단히 정리하여 서술
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(r"$$\frac{\partial E}{\partial a} = $$")
        dE_da = st.text_input(
            "",
            key="ai_step2_dE_da",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(r"$$\frac{\partial E}{\partial b} = $$")
        dE_db = st.text_input(
            "",
            key="ai_step2_dE_db",
            label_visibility="collapsed",
        )

    direction_desc = st.text_area(
        "2) 위의 결과를 바탕으로, 현재 위치에서 손실을 줄이기 위해 어떤 방향 성분이 더 필요한지 설명하고, 그에 따라 내가 선택한 이동 방향을 구체적으로 서술하시오.",
        height=100,
        placeholder="예: 두 값의 부호와 크기를 비교하여 어느 변수의 감소가 더 중요하다고 판단했는지 밝히고, 그 판단에 따라 선택한 방향을 설명하는 내용을 서술",
        key="ai_step2_direction_desc",
    )

    reflection = st.text_area(
        "3) 실제로 1 step 이동한 결과 손실값은 어떻게 변하였는가? 나의 판단과 결과가 일치하였는지 그 이유를 설명하시오.",
        height=120,
        placeholder="예: 이동 후 손실의 변화와 그 원인을 자신의 판단과 연결하여 서술",
        key="ai_step2_reflection",
    )

    st.divider()


    # (TXT/시트 저장용) 계산한 편미분 식도 함께 저장
    direction_reason = f"∂E/∂a = {dE_da.strip()}\n∂E/∂b = {dE_db.strip()}"

    col1, col2, col3 = st.columns([1, 1, 1.2], gap="small")
    with col1:
        save_clicked = st.button("✅ 제출/저장", use_container_width=True)
    with col2:
        backup_make_clicked = st.button("⬇️ TXT 백업 만들기", use_container_width=True)
    with col3:
        pass

    def _validate_step2() -> bool:
        if not dE_da.strip():
            st.error("1) ∂E/∂a 값을 입력하세요.")
            return False
        if not dE_db.strip():
            st.error("1) ∂E/∂b 값을 입력하세요.")
            return False
        if not direction_desc.strip():
            st.error("2) 방향 성분/이동 방향 설명을 입력하세요.")
            return False
        if not reflection.strip():
            st.error("3) 결과 해석을 입력하세요.")
            return False
        return True

    saved_payload = st.session_state.get(_BACKUP_STATE_KEY) or None
    payload_for_download = saved_payload if isinstance(saved_payload, dict) and saved_payload.get("student_id") == student_id else None

    if payload_for_download is None:
        payload_for_download = {
            "s": dict(s),
            "dE_da": dE_da,
            "dE_db": dE_db,
            "direction_desc": direction_desc,
            "direction_reason": direction_reason,
            "reflection": reflection,
        }

    backup_text = build_backup_text(
        payload_for_download["s"],
        payload_for_download.get("direction_desc", ""),
        payload_for_download.get("direction_reason", ""),
        payload_for_download.get("reflection", ""),
    )

    st.download_button(
        label="📄 (다운로드) 2차시 백업 TXT",
        data=backup_text.encode("utf-8-sig"),
        file_name=f"인공지능_수행평가_2차시_{student_id}.txt",
        mime="text/plain; charset=utf-8",
        use_container_width=True,
    )

    if backup_make_clicked:
        if not _validate_step2():
            st.stop()
        st.session_state[_BACKUP_STATE_KEY] = {
            "student_id": student_id,
            "s": dict(s),
            "dE_da": dE_da.strip(),
            "dE_db": dE_db.strip(),
            "direction_desc": direction_desc.strip(),
            "direction_reason": direction_reason.strip(),
            "reflection": reflection.strip(),
            "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        st.rerun()

    if save_clicked:
        if not _validate_step2():
            st.stop()

        path = s.get("path", [])
        start_a = float(s.get("start_a", path[0][0] if path else 0.0))
        start_b = float(s.get("start_b", path[0][1] if path else 0.0))
        final_a, final_b, final_e = path[-1] if path else (start_a, start_b, float(E(alpha, beta, np.array(start_a), np.array(start_b))))
        steps_used = max(0, len(path) - 1)

        s["saved_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        _set_state(s)

        try:
            from assessment.google_sheets import append_ai_step2_row  # late import

            append_ai_step2_row(
                student_id=student_id,
                alpha=alpha,
                beta=beta,
                start_a=start_a,
                start_b=start_b,
                step_size=float(s.get("step_size", STEP_SIZE)),
                dE_da=dE_da.strip(),
                dE_db=dE_db.strip(),
                direction_desc=direction_desc.strip(),
                direction_reason=direction_reason.strip(),
                result_reflection=reflection.strip(),
                final_a=float(final_a),
                final_b=float(final_b),
                steps_used=int(steps_used),
                final_E=float(final_e),
            )
            set_save_status(True, "구글시트 저장 완료")
        except Exception as e:
            set_save_status(False, f"구글시트 저장 실패: {e}")

        st.rerun()

    render_save_status()
