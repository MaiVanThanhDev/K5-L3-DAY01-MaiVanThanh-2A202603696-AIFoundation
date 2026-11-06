# Bài tập mở rộng — LLM API

## 2.1. Temperature

Đây là phân tích cơ chế, chưa phải quan sát từ bốn lần gọi API thật. Với temperature 0.0, phân phối lấy mẫu tập trung hơn; ở 0.5, 1.0 và 1.5, mô hình có xu hướng chọn cách diễn đạt và chủ đề đa dạng hơn. Temperature cao không bảo đảm câu trả lời chính xác hơn; không thể kết luận từ một phản hồi mỗi mức.

Với chatbot hỗ trợ khách hàng, tôi chọn khoảng 0–0.2 và giữ top_p mặc định để giảm biến thiên khi trả lời chính sách. Cần kết hợp dữ liệu chính sách được truy xuất, kiểm tra đầu ra và chuyển nhân viên khi thiếu thông tin; temperature thấp không ngăn được câu trả lời sai.

Có thể chạy thí nghiệm bằng đoạn sau sau khi cấu hình key qua môi trường. Không ghi key vào file:

```python
from solution import call_openai

prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
for temperature in [0.0, 0.5, 1.0, 1.5]:
    text, seconds = call_openai(prompt, temperature=temperature, top_p=1.0)
    print(temperature, text, seconds)
```

## 2.2. Chi phí

Theo đơn giá giả định trong đề, model lớn đắt gấp 10 ở cả input và output. Có 10.000 × 3 = 30.000 lần gọi/ngày, tương ứng 10,5 triệu token/ngày nếu 350 là tổng token mỗi lần. Đề chưa cho tỷ lệ input/output nên không có một mức tiền tuyệt đối duy nhất.

Ví dụ mỗi lần có 300 input + 50 output: ngày có 9 triệu input và 1,5 triệu output; terra tốn 9 × 1 + 1,5 × 5 = **16,50 USD/ngày**, luna tốn 9 × 0,1 + 1,5 × 0,5 = **1,65 USD/ngày**. Trong 30 ngày, lần lượt là 495 và 49,50 USD. Đây là phép tính theo đề, không phải hóa đơn API đã đo.

Model lớn có thể đáng tiền khi tổng hợp hồ sơ dài với nhiều ràng buộc và kiểm thử cho thấy giảm lỗi đủ để bù chi phí. Model nhỏ phù hợp phân loại SMS hoặc định tuyến yêu cầu đơn giản khi đã đạt tiêu chí chất lượng trên tập kiểm thử.

## 2.3. Streaming

Streaming hữu ích trong chat tương tác và câu trả lời dài vì người dùng đọc được phần đầu khi phần còn lại đang sinh. Nó giảm thời gian chờ cảm nhận, không bảo đảm giảm tổng thời gian sinh hay chi phí token. Non-streaming phù hợp tác vụ phân loại một từ, xử lý batch và JSON cần được nhận đủ rồi kiểm tra hợp lệ trước khi sử dụng. Với nội dung cần kiểm duyệt toàn bộ, nên đệm kết quả trước khi hiển thị.

## Bằng chứng và AI log

Code được hoàn thiện với hỗ trợ của Codex; kiểm tra tự động dùng API giả lập, không phát sinh cuộc gọi thật. Bài 2.1 chưa có bằng chứng bốn phản hồi thật, vì phiên này chưa cấu hình API key. Ước lượng trong compare_models chỉ tính output từ số từ/0,75 theo template; không bao gồm input, cached token hoặc reasoning token.

Tham số giới hạn token và reasoning đối chiếu [OpenAI API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). Cấu hình model và đơn giá trong solution giữ theo đề để tương thích bộ test; cần xác nhận quyền truy cập model trước khi chạy API thật.
