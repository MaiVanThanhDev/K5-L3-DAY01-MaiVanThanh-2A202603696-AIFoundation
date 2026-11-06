# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 800; thử thách chỉ số 2, tương ứng **đề #3**.
- **Dữ liệu:** 5.572 tin gốc; bỏ 403 tin trùng tuyệt đối còn 5.169 tin (4.516 ham, 653 spam).
- **Chia tập:** train 3.619 (467 spam), validation 775, test 775 (95 spam); SUB 40 lấy từ test, cân bằng 20 spam/20 ham.
- **Phạm vi bằng chứng:** giữ các phép đo và 10 dự đoán đã có của lần chạy Colab. Lần hoàn thiện 2026-11-06 có hỗ trợ Codex, chạy kiểm chứng cục bộ và lưu rõ nguồn; không tạo lại thời gian dự đoán và không gọi API LLM thật.

## 1. Bằng chứng

| Tệp | Nội dung và trạng thái |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Code, câu trả lời, output và biểu đồ đã lưu; bổ sung thử thách đúng đề và kết quả kiểm chứng thực tế. |
| [lab1_report.json](lab1_report.json) | SEED, đủ 10 dự đoán, 15/15 check đạt, bảng kết quả cũ; thêm `hand_calc`, `challenge_results`, `verification`, `ai_assistance`. |
| [hand_calc.jpg](hand_calc.jpg) | Ảnh tính tay đã có, thấy các bước của cả 5 phép tính; giữ nguyên ảnh. |

Mười check code đã được chạy lại, trong đó đường API dùng mock theo đề. Năm giá trị tính tay khớp ảnh và hash: `0,87; 0,50; 0,73; 0,60; 10,50`. Tương ứng: `4516/5169≈0,8737`; `σ(0)=1/2`; `F1=160/220`; `cos=4/(3√5)`; chi phí ngày `10+0,5 USD`. Accuracy nhập theo tỷ lệ 0–1, không nhập 87,37.

## 2. Kết quả chính

| Phương án | F1 test 775 | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1.000 tin |
| --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.8639 | 0.7879 | 7.93 | 4319.16 | 0.148 | 0 (API) |
| 2. BoW + MLP (deep learning) | 0.9326 | 0.9189 | 5.44 | 26353.40 | 0.277 | 0 (API) |
| 3a. Encoder (MiniLM) + kNN | 0.9091 | 0.9231 | 108.22 | 23.08 | 65.044 | 0 (API) |
| 3b. Encoder (MiniLM) + LogReg | 0.9418 | 0.9744 | 108.72 | 29.06 | 40.364 | 0 (API) |
| 3c. Encoder (MiniLM) + MLP head | 0.9622 | 0.9744 | 109.71 | 26.65 | 61.994 | 0 (API) |
| 4a. LLM nhỏ zero-shot | — | 0.9268 | 0.00 | 1.17 | 2445.881 | Chưa đo đủ* |
| 4b. LLM lớn zero-shot | — | 0.9268 | 0.00 | 0.98 | 2603.740 | Chưa đo đủ* |
| 4c. LLM nhỏ few-shot | — | 0.9189 | 0.00 | 1.00 | 2684.530 | Chưa đo đủ* |

Đây là số đo Colab đã lưu, gồm tiền xử lý từ văn bản thô khi đo p95. Encoder MiniLM đóng băng; phần train gồm encode train và học head. Chi phí API của các model local bằng 0, không có nghĩa hạ tầng/tài nguyên miễn phí. LLM thực tế dùng **Groq GPT-OSS 20B/120B**, không phải các model OpenAI nêu ở cấu hình mặc định của đề.

\* `usd_per_1k=0` ở ba hàng LLM trong JSON cũ xuất phát từ `PRICE_PER_1M` chưa có Groq. Đã bổ sung giá cho lần chạy sau, nhưng không có `usage`/reasoning token của lần cũ để khôi phục hóa đơn chính xác. Đơn giá Groq đã đối chiếu là 0,075/0,30 USD cho 20B và 0,15/0,60 USD cho 120B trên 1 triệu input/output token. [Nguồn 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b), [nguồn 120B](https://console.groq.com/docs/model/openai/gpt-oss-120b).

MLP BoW cải thiện F1 test 0,0687 so với LogReg. Trên cùng encoder, MLP head cải thiện 0,0204 so với LogReg nhưng vượt ngưỡng p95 50 ms. So sánh với LLM chỉ dùng cột SUB: hai head encoder cùng đạt 0,9744; LLM zero-shot đạt 0,9268. Chênh lệch nhỏ trên 40 tin không đủ để kết luận xếp hạng chắc chắn.

Đường Pareto trên biểu đồ `1−F1_SUB ↔ p95` gồm BoW+LogReg, BoW+MLP và MiniLM+LogReg. MiniLM+MLP head bị MiniLM+LogReg lấn át trên SUB, nhưng không bị lấn át nếu đổi tiêu chí sang F1 test đầy đủ; tập đánh giá là một phần của định nghĩa so sánh.

## 3. Dự đoán và thực tế

Giữ nguyên cả 10 dự đoán và giờ ghi ban đầu trong JSON. Ba dự đoán lệch đáng chú ý:

| Dự đoán | Giờ ghi ban đầu | Nội dung ban đầu | Thực tế | Giải thích |
| --- | --- | --- | --- | --- |
| dl_curve | 13:27:23 | val loss không giảm mãi có thể tăng, MLP sẽ kém hơn logistic regression về F1 | MLP F1 0,9326 > LogReg 0,8639; val tối ưu epoch 10 | MLP học tổ hợp phi tuyến của BoW; dữ liệu nhỏ không đồng nghĩa luôn thua mô hình tuyến tính. |
| emb_search | 13:32:27 | ham | 6/6 hàng xóm gần nhất là spam; cosine 0,523–0,546 | “complimentary holiday” gần nghĩa lời mời trúng thưởng dù tránh free/win/prize. |
| llm_models | 13:59:31 | openai/gpt-oss-120b có F1 cao hơn , chênh ~0.05 , model lớn đắt hơn gấp 4 lần , LLM với mô hình 3b ngang nhau | 20B và 120B đều F1 SUB 0,9268 | Tác vụ đơn giản và SUB chỉ 40 tin chưa cho thấy lợi ích model lớn; giá Groq đã đối chiếu gấp 2, không phải 4. |

Ngoài ra, few-shot giảm F1 từ 0,9268 xuống 0,9189, dù precision tăng lên 1 và recall giảm xuống 0,85. Không thể xác nhận dự đoán tăng chi phí 3–4 lần vì lần chạy cũ thiếu giá và không lưu token usage. Các dự đoán dùng giờ `HH:MM:SS` cũ; không bổ sung ngày hoặc tuyên bố chứng minh được thời điểm tuyệt đối ngoài dữ liệu đã lưu.

## 4. Phân tích 50 case khó

Đã chạy lại **đúng 50 case gốc** và 5 model local. F1 test thường cũng được tái lập trùng số cũ với seed/split này:

| Mô hình | F1 test 775 | F1 khó 50 | Giảm F1 | Accuracy khó 50 |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.8639 | 0.3333 | 0.5306 | 52% |
| 2. BoW + MLP | 0.9326 | 0.4211 | 0.5115 | 56% |
| 3a. Emb + kNN | 0.9091 | 0.6909 | 0.2182 | 66% |
| 3b. Emb + LogReg | 0.9418 | 0.6122 | 0.3296 | 62% |
| 3c. Emb + MLP head | 0.9622 | 0.6531 | 0.3091 | 66% |

BoW+LogReg giảm F1 nhiều nhất khi so với test đầy đủ: khoảng 0,5306. Nếu dùng SUB làm mốc, thứ tự độ giảm có thể khác; phải nói rõ tập so sánh. Bảng accuracy theo nhóm:

| Nhóm | BoW+LR | BoW+MLP | Emb+kNN | Emb+LR | Emb+MLP |
| --- | --- | --- | --- | --- | --- |
| ham_looks_spam | 100.0% | 90.0% | 70.0% | 80.0% | 80.0% |
| natural_spam | 10.0% | 10.0% | 40.0% | 20.0% | 20.0% |
| manipulation | 12.5% | 37.5% | 75.0% | 37.5% | 50.0% |
| obfuscation | 16.7% | 0.0% | 83.3% | 83.3% | 83.3% |
| prompt_injection | 66.7% | 83.3% | 83.3% | 100.0% | 100.0% |
| transactional_ham | 83.3% | 100.0% | 66.7% | 83.3% | 83.3% |
| talks_about_spam | 100.0% | 100.0% | 50.0% | 50.0% | 75.0% |

Hai case cụ thể là #12 (giọng lịch sự nhờ làm quen, dẫn tới trang hẹn hò) và #13 (nhắc xem và đặt mua ảnh sự kiện): cả 5 model local đều đoán ham dù nhãn là spam. BoW thiếu tín hiệu từ khóa; encoder gom nội dung giống hội thoại/nhắc việc nhưng không tự suy ra tính không mong muốn của lời mời. Các tin gần trùng trong train/test thường khiến điểm thường lạc quan hơn bộ kiểm tra khác phân phối này.

**Bằng chứng LLM cũ thuộc 55 case:** lần chạy cuối đã gộp bonus, nên zero-shot đúng 49/55, F1≈0,912; phòng thủ đúng 50/55, F1≈0,925. Đây không phải kết quả riêng của 50 case và không được dùng để xếp hạng trực tiếp với bảng local ở trên.

| Nhóm cũ (55 case) | Số case | LLM zero-shot đúng | LLM phòng thủ đúng |
| --- | --- | --- | --- |
| ham_looks_spam | 10 | 6/10 | 7/10 |
| natural_spam | 11 | 11/11 | 10/11 |
| manipulation | 9 | 9/9 | 9/9 |
| obfuscation | 6 | 6/6 | 6/6 |
| prompt_injection | 7 | 6/7 | 7/7 |
| transactional_ham | 7 | 6/7 | 7/7 |
| talks_about_spam | 5 | 5/5 | 4/5 |

Prompt phòng thủ giúp injection từ 6/7 lên 7/7, nhưng làm giảm một tin đúng ở natural_spam và một tin ở talks_about_spam. Không có dự đoán từng tin của cùng lần chạy 55 nên chưa xác định được chính xác case injection nào đổi nhãn. Phiên này không có key để đo lại LLM trên bộ 50. Biện pháp bổ sung: phân tách dữ liệu/chỉ dẫn, kiểm tra schema, đối chiếu rule hoặc model độc lập với yêu cầu xin OTP/chuyển tiền, chuyển bất định sang kiểm duyệt. Schema bảo đảm định dạng, không bảo đảm đúng ý định.

Với obfuscation, có thể chuẩn hóa Unicode/chữ thường/leet và ghép ký tự bị cố ý tách; cần tránh làm hỏng OTP, mã đơn hàng và tăng chặn nhầm. Bộ 50 do người viết tay phù hợp stress test và kiểm tra hồi quy, không đại diện tỷ lệ spam thực tế.

**Bonus:** giữ nguyên 5 câu đã có và bổ sung lý do riêng trong notebook; không gộp vào bộ bắt buộc:

| Case | Nhãn thật | BoW+LR | BoW+MLP | Emb+kNN | Emb+LR | Emb+MLP |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | spam | ham | ham | spam | ham | ham |
| 101 | ham | ham | ham | spam | spam | spam |
| 102 | spam | ham | ham | spam | spam | spam |
| 103 | ham | ham | ham | spam | spam | spam |
| 104 | spam | ham | ham | spam | spam | spam |

MiniLM+MLP head (F1 test thường cao nhất) sai 3/5 case: #100 xin mã xác minh bị bỏ lọt; #101 cảnh báo trích dẫn FREE và #103 OTP hợp lệ lại bị chặn. Điều này cho thấy một vector nghĩa mạnh chưa giải quyết được ý định, trích dẫn và sự khác nhau giữa “thông báo mã” với “xin mã”. kNN nhận đúng #100 nhưng nhầm #101/#103: không có model thắng ở mọi kiểu bẫy.

## 5. Thử thách cá nhân

`SEED % 6 = 800 % 6 = 2` → đề #3: đổi độ rộng **một lớp ẩn** thành 4 và 512. Bản cũ thử mạng nhiều lớp không đúng yêu cầu, nên đã thay bằng vòng so sánh cùng kiến trúc `Linear → ReLU → Linear` và thêm 64 làm mốc.

| Hidden | Số tham số | F1 test | Train CPU (s) | Epoch val tốt nhất (đếm từ 1) |
| --- | --- | --- | --- | --- |
| 4 | 8,009 | 0.9119 | 0.850 | 30 |
| 64 | 128,129 | 0.9326 | 1.081 | 10 |
| 512 | 1,025,025 | 0.9266 | 6.386 | 4 |

Công thức số tham số là `2002h+1`. Cùng seed, split, optimizer Adam, lr=0,001, batch=64 và 30 epoch; thời gian mới đo trên cùng CPU cục bộ với 4 thread. Mạng 512 nhiều tham số hơn nhưng F1 thấp hơn mốc 64 trong lần chạy này; điểm val tốt nhất xuất hiện sớm hơn. Nên chọn mạng nhỏ khi tài nguyên hạn chế và mức F1 đã đạt nhu cầu; dùng validation/early stopping để chọn cấu hình, không dùng test để chỉnh đi chỉnh lại. Đây là kết quả một seed, không khẳng định quy luật tuyệt đối.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày, tải trung bình là `10.000.000/86.400 ≈ 115,74 tin/s`; còn phải dự phòng giờ cao điểm. Tôi chọn **BoW+MLP làm đường xử lý đồng bộ ban đầu**: F1 test 0,9326, precision 1,0000, recall 0,8737, p95 0,277 ms và throughput khoảng 26.353 tin/s trên máy đã đo; chấp nhận bỏ lọt khoảng 12,6% spam ở test thường và điểm yếu trên hard cases. MiniLM+LogReg có F1 tốt hơn (0,9418), p95 40,364 ms đạt ngưỡng, nhưng chỉ 29,06 tin/s cả lô; có thể chọn làm tầng hai khi đã cấp đủ replica và kiểm thử tải. Bốn replica là mức tối thiểu lý thuyết `ceil(115,74/29,06)`, không có dự phòng và không phải cam kết SLA. MiniLM+MLP head, kNN và LLM không đạt p95 < 50 ms trong số đo này.

Thiết kế lai: BoW xử lý các tin tự tin sau khi hiệu chỉnh xác suất trên validation; tin có xác suất trung gian (ví dụ [0,3;0,7]) hoặc dấu hiệu xin OTP/chuyển tiền đi qua MiniLM/rule, rồi LLM hoặc người kiểm duyệt khi còn mâu thuẫn. LLM có p95 hơn 2 giây nên cần xử lý bất đồng bộ/hàng chờ, không đặt trên đường đồng bộ nếu mọi request đều buộc xong dưới 50 ms. Phải đo tỷ lệ routing, độ chính xác, tải và độ trễ toàn pipeline trước khi triển khai.

Theo **giả định bài tính tay**, mỗi tin 100 input + 1 output, giá 0,10/0,50 USD mỗi triệu token: 10 triệu tin/ngày × 30 ngày tốn **3.150 USD/tháng**; 1 triệu tin/ngày tốn 10,50 USD/ngày. Đây không phải chi phí Groq đã đo. Với đơn giá Groq đã đối chiếu và cùng giả định không có token suy luận/cache, 20B là 2.340 USD/tháng, 120B là 4.680 USD/tháng. GPT-OSS có reasoning nên hóa đơn thực tế cần `usage`, không thể tính đủ chỉ từ nhãn một từ. Các số `usd_per_1k=0` trong JSON gốc là do thiếu giá, không chứng minh miễn phí.

## 7. AI log và điều đã học

| Phần | Hỏi/đề nghị AI | Hỗ trợ nhận được | Kiểm chứng và giới hạn |
| --- | --- | --- | --- |
| Lần làm trước: Phần 1–3 | Gợi ý code (theo AI log đã có) | Khung làm và điểm cần lưu ý | Giữ khai báo cũ; không khẳng định các phần này hoàn toàn tự viết. |
| Lần rà soát 2026-11-06 | Đọc toàn bộ tài liệu và hoàn thiện lab | Codex đọc đề, rubric, notebook; sửa nhận xét, báo cáo và code thử thách | Chạy lại 10 check code, 5 check tính tay; đối chiếu ảnh đã có. |
| Stress test và bonus | Tách bộ 50 case khỏi 5 case cá nhân | Codex sửa cách tách và bổ sung lý do cho các case đã có | Chạy 5 model local; giữ LLM cũ trên 55 case, không giả lập kết quả API thật. |
| Thử thách #3 | Sửa lại đúng hidden=4 và 512 | Codex viết vòng so sánh 4/64/512 | Chạy cùng split/seed/30 epoch, lưu số tham số/F1/thời gian thực. |
| Side lab tùy chọn | Hoàn thiện template API và bài tập | Codex triển khai 7 hàm và phân tích câu hỏi | 19 test mock đạt; chưa thực hiện thí nghiệm temperature qua API thật. |

Lab cho phép/giới hạn AI khác nhau giữa các phần theo RULES.md; bản này khai báo đúng phạm vi AI đã tham gia, bao gồm sửa code thử thách. Không khẳng định đây là bài hoàn toàn tự làm. Người học cần tự đọc, chạy và giải thích được từng hàm trước khi nộp/vấn đáp.

Tôi hiểu thêm rằng biểu diễn tốt và head phức tạp là hai yếu tố khác nhau; thêm tham số không bảo đảm tăng F1. Accuracy, F1, p95 và throughput trả lời những câu hỏi khác nhau nên phải đọc cùng nhau. Một mô hình đạt điểm cao trên SMS cùng phân phối vẫn có thể thất bại khi ý định bị ngụy trang.

**Kiểm tra bổ sung:** bộ test repository và side lab đạt 27 test, 95 subtest. Side lab nằm ở [solution.py](../../src/optional_llm_api/solution/solution.py), câu trả lời ở [exercises.md](../../src/optional_llm_api/solution/exercises.md); thí nghiệm temperature chưa có phản hồi API thật, đã ghi rõ. Notebook không được chạy lại toàn bộ qua API trong phiên này; các kết quả gốc và kết quả kiểm chứng mới có nguồn riêng trong JSON.
