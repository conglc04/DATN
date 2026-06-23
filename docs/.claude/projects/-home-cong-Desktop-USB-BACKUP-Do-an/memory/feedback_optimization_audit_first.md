---
name: feedback-optimization-audit-first
description: Bài toán tối ưu sai → toàn bộ dự án sai. PHẢI audit formulation trước khi train.
metadata:
  type: feedback
---

Xây dựng bài toán tối ưu mà sai thì toàn bộ dự án đều sai — RL sẽ học sai.

**Why:** User nhấn mạnh: formulation (objective, constraints, reward, action space, obs) là nền tảng. Nếu code implement sai so với thiết kế, hoặc thiết kế tự mâu thuẫn, thì mọi kết quả train đều vô nghĩa.

**How to apply:** TRƯỚC KHI train bất kỳ solver nào, BẮT BUỘC audit cross-check giữa:
- docs formulation (13_methodology, 07_api_spec, 05_agent_workflow)
- code thực thi (oran_env.py, lagrangian.py, train.py, agents/, solvers/)
- tests (test coverage cho mỗi công thức)

Nếu phát hiện mâu thuẫn → DỪNG, sửa trước, KHÔNG train trên formulation sai.
