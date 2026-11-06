# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.5 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không (sử dụng nguyên bản các tham số threshold 0.35, ảnh COCO và 3 checkpoint YOLO11 nano theo hợp đồng bài thực hành)

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.511106 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về **cả bức ảnh** (image-level classification).
   - Vì: Bài toán phân loại ảnh chỉ đưa ra một nhãn duy nhất đại diện cho toàn bộ nội dung khung cảnh bức ảnh mà không có thông tin tọa độ hộp (`bbox_xyxy`) hay ranh giới pixel (`polygon_xy`) để định vị hoặc tách riêng bất kỳ một vật thể cụ thể nào. Mô hình tính toán xác suất dựa trên tổng thể đặc trưng của toàn ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm **taxonomy/checkpoint** do con người định nghĩa từ trước (cụ thể ở đây là taxonomy `ImageNet-1K` gồm 1000 lớp đi kèm checkpoint `yolo11n-cls.pt`). Mô hình không tự nghĩ ra tên lớp mới, nó chỉ tính toán phân phối điểm số (score) trên danh mục 1000 lớp cố định có sẵn này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - `class_id` (468): Là mã định danh số duy nhất giúp máy tính xử lý, lập chỉ mục và truy vấn dữ liệu chính xác, tránh sai sót do định dạng văn bản (chữ hoa/thường, khoảng trắng).
   - `class_name` ("cab"): Giúp con người (annotator, reviewer, engineer) đọc và hiểu ngữ nghĩa trực quan.
   - `taxonomy_name` ("ImageNet-1K"): Xác định rõ không gian nhãn và ngữ cảnh tham chiếu, vì cùng một tên lớp hoặc ID ở các taxonomy khác nhau (như ImageNet vs COCO vs OpenImages) có thể mang định nghĩa, phạm vi ngữ nghĩa và ranh giới rất khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần quy định rõ:
     1) Nguyên tắc ưu tiên chủ thể chính (dominant object) chiếm diện tích lớn nhất hoặc nằm ở vị trí tiền cảnh/trung tâm bức ảnh;
     2) Bảng thứ tự ưu tiên giữa các loại đối tượng (ví dụ: ưu tiên phương tiện giao thông hơn biển báo/cảnh nền);
     3) Trường hợp không có chủ thể nào áp đảo rõ rệt, quy định chuyển sang phân loại đa nhãn (multi-label), gán nhãn cảnh quan chung (`street scene`), hoặc gắn cờ escalation để reviewer quyết định.
6. Vì sao không được lấy `score` của model làm ground truth?
   - Vì `score` của mô hình (ở đây là 0.511106) chỉ là giá trị tự tin thống kê (confidence) xuất ra từ hàm Softmax dựa trên các trọng số đã học. Con số này không bảo đảm tính đúng đắn khách quan thực tế và chưa được con người kiểm chứng. Ground truth phải là nhãn sự thật do con người tạo lập và kiểm soát chất lượng nghiêm ngặt theo guideline chuẩn của dự án.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912559 ; `bbox_xyxy`=[385.34, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Hộp bao quanh người đang đứng nấu ăn ở góc bên phải của gian bếp:
     - Góc trên-trái: tọa độ x = 385.34 pixel, y = 69.24 pixel (tính từ gốc (0, 0) ở góc trên cùng bên trái của bức ảnh).
     - Góc dưới-phải: tọa độ x = 498.92 pixel, y = 348.92 pixel.
     - Chiều rộng: 113.58 pixel (`x_max - x_min`).
     - Chiều cao: 279.68 pixel (`y_max - y_min`).
     - Hộp ôm trọn từ đỉnh đầu của người cho tới phần thân và đầu gối (khu vực chân bị che bởi kệ bếp).
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Ngưỡng 0.20: có **17** predictions (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`, `bowl`, `bowl`, `cup`, `cup`, `bowl`, `spoon`, `potted plant`, `spoon`, `dining table`, `spoon`, `bottle`).
   - Ngưỡng 0.60: có **6** predictions (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Khi hạ ngưỡng (từ 0.35 xuống 0.20), số lượng prediction tăng vọt từ 11 lên 17. Reviewer phải kiểm tra khối lượng hộp lớn hơn rất nhiều, đồng thời phải tốn nhiều thời gian và thao tác để xóa bỏ các hộp dự đoán sai (false positives), các hộp nhiễu, hoặc các hộp bao trùng lặp lên cùng một vật thể.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - **Quy tắc:** "Hộp bao phải ôm sát ranh giới ngoài cùng của vật thể (tight bounding box), khoảng hở giữa 4 cạnh của hộp và viền ngoài cùng của vật thể không được vượt quá 2 pixel; không để lọt khoảng trống nền lớn ở các cạnh và tuyệt đối không cắt lẹm vào các bộ phận nhìn thấy được của vật thể."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - **Vật quan sát được:** Phần thân dưới / chân của người (`person`) bị che khuất (occluded) bởi quầy kệ bếp phía trước; hoặc chiếc lò nướng (`oven`) ở góc dưới bên phải bị mép ảnh cắt ngang (truncated).
   - **Quyết định:** Viết thành **quy tắc trong guideline**.
   - **Vì sao:** Hiện tượng che khuất (occlusion) và cắt mép ảnh (truncation) là tình huống xuất hiện thường xuyên trong ảnh đời thực. Guideline hoàn toàn có thể chuẩn hóa quy tắc: *"Chỉ đóng khung phần nhìn thấy được (visible bounding box), gắn cờ thuộc tính `is_occluded = True` hoặc `is_truncated = True`, và chỉ gán nhãn nếu phần nhìn thấy được >= 15% diện tích ước lượng của vật thể"*. Trường hợp này không có sự mập mờ về nhận thức (ai nhìn vào cũng biết đó là người đang đứng sau kệ), nên không cần làm phiền Lab Coach mà cần chuẩn hóa thành quy tắc trong guideline để toàn đội ngũ annotator áp dụng đồng nhất.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899351 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Polygon cung cấp ranh giới hình học chính xác ở cấp độ từng pixel (pixel-level mask) ôm theo viền thực của vật thể. Trong khi bounding box là hình chữ nhật bao gồm cả pixel nền trống và các vật thể lân cận lọt vào 4 góc, polygon loại bỏ hoàn toàn phần nền trống, phân tách rõ ràng các đường cong cơ thể, cánh tay, vai, nếp gấp quần áo và phần tiếp giáp với đồ vật xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - `instance_id` (như `kitchen-001`, `kitchen-002`) dùng để định danh và phân biệt riêng từng cá thể độc lập trong cùng một bức ảnh, ngay cả khi chúng thuộc cùng một lớp (ví dụ phân biệt người thứ nhất và người thứ hai, hoặc chiếc bát này với chiếc bát kia trong bếp).
   - Câu nói này có nghĩa:
     - Nó **không phải class ID**: class ID là mã chung của loại đối tượng (ví dụ class 0 là person, class 45 là bowl); nhiều đối tượng có thể chung một class ID, nhưng mỗi đối tượng bắt buộc phải có một `instance_id` riêng biệt.
     - Nó **không phải tracking ID**: tracking ID là mã định danh duy trì sự liên tục của cùng một đối tượng xuyên suốt qua chuỗi khung hình video thời gian thực; trong khi `instance_id` ở đây chỉ có giá trị cục bộ trong bức ảnh đơn lẻ đang xét.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - **Quy tắc:** "Đường biên đa giác (polygon) phải bám khít đường viền thực tế của vật thể với sai số không quá 1-2 pixel; tại các khúc cong tròn hoặc chi tiết phức tạp phải tăng mật độ điểm để phản ánh đúng hình dáng hình học, tuyệt đối không tạo các đoạn thẳng cắt góc (polygon shortcutting), không để lấn sang phần nền hoặc vật thể kế bên."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - **Vị trí quan sát được:** Vùng tiếp xúc giữa đáy các chiếc bát (`bowl`) với bề mặt kệ bếp, hoặc vùng tiếp xúc giữa bàn tay của người (`person`) với mặt bếp.
   - **Quyết định:** Viết thành **quy tắc trong guideline**.
   - **Vì sao:** Việc các vật thể đặt trên mặt phẳng chạm nhau là quy luật vật lý hiển nhiên trong mọi gian bếp. Guideline hoàn toàn có thể định nghĩa nguyên tắc chuẩn: *"Áp dụng nguyên tắc không chồng lấn mask (non-overlapping masks) – đường biên tiếp xúc thuộc về vật thể nằm ở lớp trên (cái bát); tuyệt đối không để polygon của bát ăn lẹm vào kệ bếp và ngược lại"*. Chỉ cần dùng escalation khi hai vật thể đồng màu, chất liệu phản quang gây lóa sáng khiến mắt thường không thể phân biệt ranh giới vật lý giữa chúng.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp ảnh (`class_id`, `class_name`, `taxonomy_name`), nhãn đơn hoặc đa nhãn, không có tọa độ. | Ảnh `traffic` có nhiều loại phương tiện hỗn hợp (taxi, xe bus, xe van, người đi đường); mô hình dự đoán `cab` chỉ đạt score ~0.5111. | Đọc kỹ thứ tự ưu tiên trong guideline (chọn chủ thể chính/chiếm ưu thế ở tiền cảnh, hoặc gán đa nhãn nếu guideline cho phép); nếu không phân định được, gửi escalation. | Kiểm tra nhãn được chọn có tuân thủ đúng quy tắc ưu tiên chủ thể trong guideline không; rà soát lỗi nhầm lẫn giữa các class tương đồng (như cab vs minibus). |
| Phát hiện vật thể | Danh sách các hộp giới hạn `[x_min, y_min, x_max, y_max]` kèm `class_id` cho từng đối tượng nhìn thấy. | Bỏ sót các vật thể nhỏ/bị che khuất (như thìa, cốc); hoặc sinh ra hai hộp bao trùng lặp cho cùng một lò nướng (`oven`) ở các mức threshold khác nhau. | Zoom cận cảnh để quét sạch các đối tượng trong ảnh; vẽ bounding box ôm khít 4 biên ngoài cùng của vật thể; gắn cờ `occluded`/`truncated` đúng quy định. | Soi độ khít (tightness) của 4 cạnh box; rà soát các vật thể bị sót (false negatives) và các box vẽ trùng lặp/thừa thãi (false positives). |
| Instance segmentation | Tập hợp tọa độ đỉnh `polygon_xy` (hoặc binary mask RLE) kèm `instance_id`, `class_id` cho từng cá thể. | Đường biên polygon bị răng cưa, thiếu điểm tại các đường cong; hoặc lấn sang bề mặt kệ bếp/nền gạch; vùng tiếp xúc giữa các chiếc bát dính vào nhau. | Zoom lớn khi chấm điểm đa giác; đảm bảo đủ mật độ điểm tại các khúc cua; tuân thủ nguyên tắc không để mask chồng lấn; tách riêng từng cá thể bằng `instance_id` riêng. | Bật lớp hiển thị mask overlay để soi độ bám biên (edge alignment); kiểm tra lỗi ăn lẹm/hở viền; đảm bảo tính phân tách độc lập giữa các instance kề nhau. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Tuyệt đối tuân thủ quyền riêng tư và an toàn thông tin: Không bao giờ tải lên, lưu trữ hoặc commit công khai lên GitHub/Colab các dữ liệu cá nhân định danh (PII - khuôn mặt người thật chưa che mờ, biển số xe, thông tin liên lạc cá nhân, CCCD) cũng như dữ liệu nội bộ/khách hàng chưa được cấp phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Tôi sẽ dừng công việc ngay lập tức, cô lập dữ liệu và báo cáo cho Giảng viên hướng dẫn (GV) / Lab Coach / Lead phụ trách dự án để được xử lý theo quy trình.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
