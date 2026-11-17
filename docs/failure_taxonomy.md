# Failure Taxonomy (LogiX-VN)

Danh mục phân loại các lỗi (Failure Modes) thường gặp trong hệ thống tác vụ tự động và agent của LogiX-VN:

- **FM-01**: **Approval Bypass** – Tác vụ yêu cầu phê duyệt thủ công/cấp cao nhưng agent tự ý thực thi qua mặt (bypass).
- **FM-02**: **Incorrect Refund Amount** – Số tiền hoàn trả (refund) tính sai hoặc không khớp với hạn mức/ledger quy định.
- **FM-03**: **State Drift / Inconsistency** – Trạng thái ghi nhận trong cơ sở dữ liệu/ledger mâu thuẫn với phản hồi trả về cho người dùng.
- **FM-04**: **Unauthorized Tool Execution** – Gọi công cụ/API ngoài phạm vi phân quyền hoặc chưa được xác thực.
- **FM-05**: **Idempotency Violation** – Xử lý giao dịch trùng lặp (ví dụ: hoàn tiền 2 lần cho cùng một claim).
- **FM-06**: **Secret / PII Leak** – Làm lộ thông tin nhạy cảm, token, hoặc dữ liệu cá nhân trong trace/phản hồi.
- **FM-07**: **Flaky / Non-deterministic Failure** – Lỗi không ổn định, phụ thuộc vào seed ngẫu nhiên hoặc làm tròn số ở 1/5 seed.
- **FM-08**: **Reply-versus-Ledger Mismatch** – Phản hồi văn bản của agent khẳng định đã thực hiện xong giao dịch, nhưng trạng thái ledger thực tế chưa ghi nhận thành công.