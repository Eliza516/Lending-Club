# Mô hình này được xây dựng như thế nào — Góc nhìn của một Data Scientist ở bộ phận Credit Risk

> **Lưu ý về ngôn ngữ.** Đây là bản dịch tiếng Việt của `docs/methodology_walkthrough.md`,
> dùng để đọc và hiểu nội bộ. Brief môn học yêu cầu báo cáo nộp **100% tiếng Anh**, nên
> **bản tiếng Anh mới là bản chính thức** để trích vào report. Hai bản phải được cập nhật
> cùng nhau.

Tường thuật ngôi thứ nhất về lý do đằng sau mọi quyết định trong project: tôi đã nhìn gì,
rút ra kết luận gì, làm gì, và mỗi lựa chọn phải trả giá ra sao.

Đây là tài liệu trả lời **"tại sao"**. Các notebook trả lời **"cái gì"**. Hãy đọc tài liệu
này trước nếu bạn cần bảo vệ, mở rộng hoặc audit mô hình.

---

## Phase 0 — Trước khi mở file dữ liệu

Với một dataset nổi tiếng, cám dỗ thường thấy là load lên rồi vẽ biểu đồ ngay. Tôi không
làm vậy, vì những câu hỏi quyết định mô hình tín dụng có dùng được hay không **không thể
trả lời từ dữ liệu**.

**Những gì phải chốt trước:**

| Câu hỏi | Câu trả lời của tôi | Vì sao phải đến trước |
|---|---|---|
| Ai hành động dựa trên điểm số? | Ra quyết định tín dụng: duyệt/từ chối, bậc lãi suất, hạn mức | Nếu không quyết định nào thay đổi thì dừng — không có project |
| Dự đoán cái gì, vào lúc nào? | `P(charged off)` tại thời điểm nộp đơn | Điều này quyết định cột nào hợp lệ. Mọi thứ phía sau phụ thuộc vào nó |
| Sai thì tốn kém thế nào? | FN (bỏ sót default) ≫ FP (từ chối khách tốt) | Quyết định metric và ngưỡng cắt, không phải thuật toán |
| Hiện đang có sẵn cái gì? | **`grade` / `sub_grade` của chính Lending Club** | Đây là incumbent. Là thứ tôi bắt buộc phải vượt qua |

Câu thứ tư là câu hầu hết project bỏ qua, và bỏ qua nó chính là lý do người ta tự hào với
AUC 0.73 mà chưa bao giờ kiểm tra rằng thang grade 7 bậc có sẵn của nền tảng **tự nó đã đạt
~0.68**. **Một mô hình chỉ xứng đáng với độ phức tạp của nó khi so với cái quy tắc mà nó
thay thế.**

**Quyết định:** chốt hai baseline trước khi mô hình hoá — B0 majority class (sàn kiểm tra
tính hợp lý), B1 `sub_grade` dùng trực tiếp làm điểm rủi ro (vạch thật sự cần vượt).

Tôi cũng ghi tỷ lệ chi phí là `FN:FP = 4:1` và **đánh dấu rõ đây là giá trị tạm**. Gọi một
con số phỏng đoán đúng tên của nó là cách duy nhất ngăn nó lặng lẽ biến thành "phát hiện".

---

## Phase 1 — Tiếp xúc đầu tiên: đọc dữ liệu bằng con mắt risk analyst

File gốc có ~2.26 triệu dòng × 151 cột, giai đoạn 2007–2018 Q4. *(Kích thước và số dòng
resolved nêu dưới đây là theo tài liệu của file Kaggle thật và được đối chiếu với các
project khảo sát trong `docs/related_work.md` — project này **chưa** chạy trên file đó; xem
Phase 16.)* Ba thứ tôi kiểm tra trước mọi thứ khác, theo đúng thứ tự này.

**1. Một dòng là gì?** Một **đơn vay đã được giải ngân**. Không phải một người vay (có
người vay nhiều lần, và không có khoá định danh ổn định để group). Không phải một đơn ứng
tuyển (hồ sơ bị từ chối nằm ở file khác, không có kết quả). Chỉ một câu này đã hàm ý ngay
**selection bias "chỉ có hồ sơ được duyệt"** — thứ giới hạn mọi khẳng định tôi đưa ra về
sau.

**2. Trục thời gian là gì?** `issue_d`, độ phân giải theo tháng, 2007–2018. Khối lượng giải
ngân tăng vài bậc độ lớn qua cửa sổ thời gian. Điều này lập tức cho tôi biết **random split
sẽ sai** — chi tiết ở Phase 4.

**3. Cột kết quả thực sự chứa gì?** `loan_status` có khoảng bảy giá trị, không phải hai.
`Current`, `Late (31–120 days)`, `In Grace Period`, `Issued` là **chưa ngã ngũ** — kết quả
chưa biết. Chỉ `Fully Paid`, `Charged Off` và `Default` là trạng thái cuối.

Quan sát thứ ba là nơi quyết định thực sự đầu tiên nằm ở đó.

---

## Phase 2 — Phân loại cột: phép thử thời điểm (timing test)

151 cột, và phần lớn là **độc**. Tôi áp một câu hỏi duy nhất lên từng cột một:

> *Vào ngày đơn vay đến, trường dữ liệu này đã có giá trị chưa?*

Câu trả lời chia file thành hai nhóm.

**Hợp lệ (~30 cột):** yêu cầu khoản vay (số tiền, kỳ hạn, mục đích), thông tin người vay
(thu nhập, việc làm, nhà ở, bang), và bản tra cứu tín dụng (dải FICO, DTI, tỷ lệ sử dụng
hạn mức, số tài khoản mở, lịch sử trễ hạn, độ dài hồ sơ tín dụng).

**Độc (~40 cột):** bất cứ thứ gì chỉ tồn tại **vì tiền đã được cho vay rồi**.

| Nhóm cột | Vì sao chí mạng |
|---|---|
| `recoveries`, `collection_recovery_fee` | Chỉ khác 0 **sau khi** đã charge-off. Đây là nhãn đội mũ nguỵ trang |
| `total_pymnt*`, `total_rec_*` | Khoản vay default thì trả ít hơn. Đây là nhãn dưới dạng biến liên tục |
| `last_pymnt_*`, `next_pymnt_d`, `out_prncp*` | Hành vi trả nợ — tức chính thứ đang cần dự đoán |
| `last_fico_range_*` | FICO **tra lại trong lúc đang vay**. Nó dịch chuyển *cùng với* việc default |
| `hardship_*`, `settlement_*`, `debt_settlement_flag*` | Chỉ có giá trị với khoản vay đang gặp khó khăn |
| `funded_amnt*` | Sau quyết định. Hãy dùng `loan_amnt` — số tiền người vay *đề nghị* |

Đây chính là kiểu lỗi sinh ra những notebook AUC 0.99 trên dataset này. Mô hình không dự
đoán default; nó đang **đọc một bản sao nguỵ trang của đáp án**.

**Quyết định:** tôi giới hạn `read_csv(usecols=...)` theo danh sách hợp lệ — để cột độc
**không bao giờ vào memory** — và *đồng thời* vẫn giữ một bước drop `LEAKAGE_COLUMNS` tường
minh phía sau, như lớp phòng vệ chống việc ai đó về sau mở rộng danh sách. Hai lớp, vì lỗi
này âm thầm và đắt.

Tôi cũng loại `emp_title` và `title`: văn bản tự do, hàng chục nghìn giá trị khác nhau,
không có cấu trúc thứ tự, và là phương tiện tự nhiên cho target encoding vô tình.

**Quy tắc kiểm tra tôi ghi hẳn vào repo:** dải hợp lý trên dataset này là ROC-AUC ≈
**0.68–0.72**. Bất cứ con số nào gần 0.99 nghĩa là một cột độc đã lọt. Các công trình đã
công bố trải từ 0.678–0.735 (`docs/related_work.md`), xác nhận dải này.

---

## Phase 3 — Định nghĩa nhãn, và cái giá phải trả

Chỉ giữ trạng thái cuối: `Fully Paid` → 0, `Charged Off`/`Default` → 1. Bỏ phần còn lại.

Đây là việc bắt buộc — không thể huấn luyện trên một kết quả chưa xảy ra. Nhưng nó **không
miễn phí**, và tôi bắt mình phải viết hoá đơn ra:

Một khoản vay 36 tháng giải ngân cuối 2018 **không thể** đã `Fully Paid` tại thời điểm chụp
dữ liệu 2018 Q4. Nó chỉ có thể ngã ngũ nếu ngã ngũ **sớm** — mà ngã ngũ sớm thì thiên lệch
về phía *charge-off*. Vậy nên việc lọc theo trạng thái cuối **âm thầm làm giàu giai đoạn
gần đây bằng các ca default**.

**Hệ quả:** tỷ lệ default quan sát được ở giai đoạn cuối **không phải** tỷ lệ default thật.
Tôi thấy điều này trực tiếp trên biểu đồ tỷ lệ default theo thời gian — nó bẻ lên ở rìa
phải. Một phần là drift thật; một phần là hiện vật thống kê này. Tôi không thể tách bạch
hai phần đó.

**Quyết định:** giữ bộ lọc, ghi nhận bias này thành một limitation có tên hẳn hoi, và
**không bao giờ** trích tỷ lệ default của giai đoạn test như thể đó là tỷ lệ thật của nhóm.
(Phase 16 quay lại chuyện này — đây là chỗ duy nhất một project đã công bố làm tốt hơn tôi.)

Trên file thật, bước này còn lại ~1.3 triệu khoản vay đã ngã ngũ, tỷ lệ default ~20%, nhất
quán giữa các project được khảo sát.

---

## Phase 4 — Đóng băng split **trước khi** nhìn bất cứ thứ gì

Đây là bước mà thứ tự quan trọng nhất và việc vi phạm khó phát hiện nhất.

**Vì sao out-of-time chứ không random.** Dự đoán default là bài toán **dự báo**. Mô hình sẽ
chấm điểm người nộp đơn **quý sau**. Suốt 2007–2018, khối lượng, cơ cấu sản phẩm, chính sách
tín dụng và bối cảnh vĩ mô của nền tảng đều dịch chuyển. Random split cho phép mô hình học
trên khoản vay 2018 để dự đoán khoản vay 2015 — thông tin mà không scorecard triển khai thật
nào có được. Nó không làm mô hình tốt hơn; nó làm **phép đo** đẹp lên.

**Quyết định:** sắp xếp theo `issue_d`, train trên mọi thứ trước `2016-01-01`, test trên mọi
thứ từ ngày đó trở đi. Đóng băng.

Ba ghi chú thiết kế:

- **Một mốc ngày tốt hơn một seed khi làm giao thức chung.** Môn học yêu cầu các nhóm cùng
  dataset dùng chung một split. Mốc ngày không phụ thuộc RNG của thư viện nào, nên tái lập
  chính xác qua mọi cách cài đặt và mọi phiên bản. `train_test_split(random_state=42)` thì
  không.
- **Tỷ lệ thu được là ~50/50, không phải 80/20.** Tăng trưởng khối lượng khiến 2016 rơi vào
  giữa phần dữ liệu đã ngã ngũ. Tôi **không** dịch mốc cắt để có tỷ lệ đẹp hơn — chọn split
  bằng cách nhìn vào kết quả nó tạo ra chính là chọn theo kết quả.
- **Tôi vẫn tính random split**, nhưng chỉ như một con số tham chiếu, để định lượng xem nó
  mua về bao nhiêu phần lạc quan ảo. Nó không bao giờ là kết quả chính.

**Vì sao phải trước EDA.** Một analyst đã nghiên cứu giai đoạn test thì đã làm rò rỉ nó rồi
— không qua code, mà qua chính những lựa chọn feature và mô hình của anh ta. **Không metric
nào phát hiện được điều này.** Nên split được quyết định trước, và mọi khám phá sau đó chỉ
nhìn thấy dòng dữ liệu train.

Tôi ghi `split_manifest.json` lưu mốc cắt, số dòng mỗi bên, khoảng thời gian, cân bằng lớp
và một hash của chỉ số test, để split có thể audit được và nhóm khác chứng minh được họ đã
tái lập đúng.

> **Một bug tôi bắt được ở đây.** Ban đầu tôi ghi manifest ngay sau khi split — nhưng các
> bước sau vẫn còn xoá dòng (trùng lặp, giá trị bất khả thi). Với file thật, số liệu trong
> manifest sẽ lệch với dữ liệu xuất ra, và assertion ở notebook sau sẽ fail. Nó qua được
> trên mẫu synthetic chỉ vì không có dòng nào bị xoá. Đã sửa: manifest được ghi **sau** khi
> lọc xong toàn bộ dòng, tính trên tập dòng cuối cùng.

---

## Phase 5 — Làm sạch: những gì dữ liệu nói mà không thể là sự thật

Ở đây có ba việc khác hẳn nhau, và gộp chúng lại chính là chỗ lỗi ẩn mình.

**(a) Parsing.** File gốc lưu `int_rate` là `"13.56%"`, `term` là `" 36 months"`,
`emp_length` là `"10+ years"`, ngày tháng là `"Dec-2015"`. Thuần tuý xử lý chuỗi — theo
từng dòng, tất định, không học thống kê nào. An toàn trên toàn bộ frame.

**(b) Kiểm tra logic.** Trước khi điền khuyết bất cứ gì, tôi kiểm tra dữ liệu có mô tả một
thế giới khả dĩ hay không:

| Kiểm tra | Vi phạm nghĩa là gì |
|---|---|
| `earliest_cr_line` sau `issue_d` | Hồ sơ tín dụng mở sau khoản vay. Bất khả thi |
| `fico_range_high` < `fico_range_low` | Dải FICO ngược. Bất khả thi |
| `annual_inc` ≤ 0 | Bất khả thi với một hồ sơ đã được duyệt |
| `open_acc` > `total_acc` | Khó tin, nhưng không bất khả thi — độ trễ báo cáo của bureau |

Tôi **báo cáo số lượng trước khi hành động**, vì con số chính là chẩn đoán: vài dòng là lỗi
dữ liệu; hàng nghìn dòng nghĩa là **code parse của tôi sai**, chứ không phải Lending Club
công bố hàng nghìn khoản vay bất khả thi. Những vi phạm bất khả thi cứng thì bị xoá — không
có cách trung thực nào để bịa ra ngày mở hồ sơ tín dụng, và điền khuyết lên nó sẽ che mất
một tín hiệu về chất lượng dữ liệu.

**(c) Ngoại lai — và cái bẫy.** Nước đi hiển nhiên là cắt ở phân vị 1%/99%. **Đó là
leakage.** Phân vị là một **thống kê tính từ dữ liệu**; dùng nó trước khi split cho phép
phân phối thu nhập của giai đoạn test định hình cách các dòng train bị biến đổi. Thay vào
đó tôi cắt ở **biên miền cố định** (`annual_inc` ≤ 1.5 triệu USD, `dti` ≤ 60) — những giới
hạn đến từ tính hợp lý nghiệp vụ, không phải từ chỗ một phân vị tình cờ rơi vào.

Cùng logic đó buộc `revol_util` phải chia bucket theo **cạnh cố định** (0/25/50/75/100) chứ
không theo phân vị.

**(d) Chuẩn hoá nhãn.** `home_ownership` chứa `ANY`, `NONE` và `OTHER` — ba nhãn cho cùng
một nhóm "còn lại", là hiện vật của việc biểu mẫu thay đổi qua các năm. Để nguyên thì chúng
thành ba dummy thưa cùng nghĩa, và tệ hơn, **tần suất tương đối của chúng dịch chuyển theo
thời gian**, nên encoder sẽ học một bộ từ vựng cho giai đoạn train khác với những gì giai
đoạn test chứa. Đã gộp làm một.

---

## Phase 6 — Thiếu dữ liệu: ô trống có đang nói điều gì không?

Nước đi tiêu chuẩn là `fillna(median)` rồi đi tiếp. Ở đây việc đó **phá huỷ thông tin**.

Tôi chia các cột thiếu thành hai loại:

**Thiếu có cấu trúc — giá trị không tồn tại.** `mths_since_last_delinq` trống với khoảng
một nửa số người nộp đơn, và ô trống đó nghĩa là **"người này chưa từng trễ hạn"** — dữ kiện
**bảo vệ** mạnh nhất trong cả file. Điền trung vị vào đó lại nói rằng "lần trễ hạn gần nhất
của họ cách đây một khoảng thời gian trung bình" — tức là **ngược hoàn toàn**.

**Không được ghi nhận — giá trị có tồn tại nhưng không được thu thập.** `mort_acc`,
`pub_rec_bankruptcies` thiếu ở các vintage cũ vì khi đó trường dữ liệu bureau chưa trả về.
Ô trống nói về **thời kỳ báo cáo**, không nói về người vay.

**Quyết định:** với các cột thiếu có cấu trúc, thêm một cột chỉ báo `_was_missing` tường
minh **trước** khi điền khuyết. Imputer sau đó điền giá trị; cột chỉ báo giữ lại sự kiện
"đã từng trống", và mô hình có thể học xem sự vắng mặt đó nghĩa là gì.

Tôi **kiểm chứng** thay vì giả định: trên dữ liệu train, nhóm có `mths_since_last_delinq`
trống default ở mức **16.2%** so với **18.1%** ở nhóm có giá trị. Ô trống mang tính bảo vệ,
đúng như lập luận nghiệp vụ dự đoán. Nếu hai tỷ lệ bằng nhau, tôi đã bỏ cột chỉ báo đi — một
cột không thêm thông tin thì chỉ thêm phương sai.

**Sàng lọc cột.** Tôi bỏ các cột thiếu trên 60% số dòng — nhưng ngưỡng được đo **chỉ trên
dòng train**, rồi áp cho cả hai bên. Tỷ lệ thiếu là một **thống kê**, và đo nó trên toàn bộ
frame tức là cho giai đoạn test bỏ phiếu về việc mô hình được dùng cột nào.

---

## Phase 7 — Feature: mã hoá những gì một cán bộ thẩm định thực sự suy nghĩ

Một cặp cột thô không tự cho mô hình tuyến tính một tỷ lệ. Các feature tôi xây là những đại
lượng mà một cán bộ tín dụng sẽ tự tính bằng tay:

| Feature | Lý do |
|---|---|
| `loan_to_income` | Đòn bẩy — thước đo khả năng chi trả dễ diễn giải nhất |
| `installment_to_income` | Gánh nặng trả nợ của **chính khoản vay này**, thứ `dti` không tính |
| `credit_history_years` | Hồ sơ mỏng thì rủi ro hơn. (Dùng `issue_d` làm proxy cho ngày nộp đơn — xem Phase 16) |
| `fico_avg` | Bureau báo cáo một dải; điểm giữa là đại lượng vô hướng dùng được |
| `is_36_month` | Khoản vay 60 tháng rủi ro hơn về mặt cấu trúc |
| `grade_ordinal`, `sub_grade_ordinal` | **Thứ tự, không one-hot.** Các grade có thứ tự và tỷ lệ default đơn điệu theo chúng — one-hot sẽ vứt bỏ đúng cấu trúc đó |

Tất cả đều theo từng dòng, nên đều an toàn trên toàn bộ frame.

**Thứ tôi cố tình KHÔNG xây: target encoding.** Thay `addr_state` bằng tỷ lệ default trung
bình của bang đó là một kỹ thuật mạnh và là phép biến đổi **nguy hiểm nhất** có thể dùng ở
đây. Tính trên chính các dòng mà mô hình học, mỗi dòng đã **nhìn trộm một phần đáp án của
chính nó** — một bang chỉ có một khoản vay sẽ nhận encoding đúng bằng kết quả của khoản vay
đó. Nếu về sau có thêm, dạng an toàn duy nhất là `TargetEncoder` **bên trong `Pipeline`**,
để nó fit theo kiểu out-of-fold. Tôi ghi chú điều này trong notebook thay vì để lại một cái
bẫy.

---

## Phase 8 — Làm cho leakage **bất khả thi về mặt cấu trúc**, không chỉ là "tránh"

Điền khuyết học một trung vị. Scaling học một trung bình và độ lệch chuẩn. One-hot encoding
học một bộ từ vựng danh mục. Mỗi thứ trong đó nếu fit trên toàn bộ frame đều là nhiễm bẩn,
và **không metric nào lộ ra điều đó** — điểm số chỉ lặng lẽ đẹp hơn mức đáng có.

Phản mẫu:

```python
preprocessor.fit(X)                       # <- nhìn thấy dòng test
X_train, X_test = train_test_split(X)     # <- đã quá muộn
```

**Quyết định:** mọi phép biến đổi có học tham số đều nằm trong một `ColumnTransformer` bên
trong một `Pipeline`. `X_test` khi đó chỉ đến được với nó qua `.transform()`, không bao giờ
qua `.fit()`. Điều này biến hành vi đúng thành **cấu trúc**, chứ không phụ thuộc vào kỷ luật
của tôi vào một chiều thứ Sáu. Nó cũng có nghĩa cross-validation **fit lại** phần tiền xử lý
bên trong từng fold thay vì một lần trên toàn bộ dữ liệu train.

**Cách tôi kiểm chứng thay vì tin tưởng:** tôi so trung vị mà imputer học được với (a) trung
vị của phần fit slice và (b) trung vị của toàn bộ frame. Chúng **khớp (a) và khác (b)** —
chênh lệch lớn nhất 275 đơn vị ở một cột. Nếu tiền xử lý bị nhiễm bẩn, chúng đã khớp (b).

---

## Phase 9 — Baseline trước đã

Trước mọi mô hình:

| Baseline | ROC-AUC trên test | Cách đọc |
|---|---|---|
| B0 majority class | 0.5000, PR-AUC = prevalence | Cái sàn. Vượt nó không chứng minh được gì |
| **B1 `sub_grade` (incumbent)** | **0.6164** | Vạch thật sự |

B1 **không cần fit gì cả** — `sub_grade` vốn đã là một thứ hạng rủi ro có thứ tự, và mọi
metric dựa trên xếp hạng (AUC, PR-AUC, KS) đều dùng trực tiếp được.

Có con số này **trước** khi mô hình hoá làm thay đổi cách đọc mọi kết quả về sau. Câu hỏi
không còn là "AUC của tôi có tốt không?" mà thành "**AUC của tôi có tốt hơn cái quy tắc nền
tảng đang chạy không?**"

---

## Phase 10 — Mô hình, từ đơn giản đến phức tạp, và cách đọc overfitting

Bốn mô hình, mỗi cái là một `Pipeline` hoàn chỉnh: logistic regression (scorecard baseline
có thể audit), random forest, HistGradientBoosting, XGBoost.

Cross-validation dùng `TimeSeriesSplit` trên các dòng train đã sắp theo thời gian — **không
bao giờ** dùng fold xáo trộn, vì điều đó sẽ tái lập đúng cái look-ahead mà split OOT đã loại
bỏ.

Tôi yêu cầu lấy cả điểm trên train-fold bên cạnh điểm validation, và chính bảng đó cho ra
phát hiện hữu ích nhất của project:

| Mô hình | Train AUC | CV AUC | Gap |
|---|---|---|---|
| Logistic Regression | 0.675 | 0.632 | **+0.044** |
| Random Forest | 0.785 | 0.644 | +0.140 |
| HistGradientBoosting | 0.938 | 0.606 | **+0.332** |
| XGBoost | 0.992 | 0.596 | **+0.396** |

Các ensemble đang **học thuộc lòng**. XGBoost khớp gần như hoàn hảo trên train-fold và tổng
quát hoá tệ nhất. Không có cột này thì thứ hạng cuối trông như ngẫu nhiên; có nó thì kết quả
trở nên hiển nhiên và giải thích được.

**Tuning để sau cùng**, có chủ ý — sau khi đã chốt so sánh, chấm bằng PR-AUC với
`TimeSeriesSplit`. Tuning trước khi so sánh là đo công sức tìm kiếm, không phải đo chất lượng
mô hình.

**Kết quả test cuối (mẫu synthetic — xem cảnh báo bên dưới):**

| Mô hình | ROC-AUC | PR-AUC | so với B1 |
|---|---|---|---|
| Logistic Regression | 0.6421 | 0.2920 | **+0.0257** |
| Random Forest | 0.6387 | 0.2834 | +0.0223 |
| B1 `sub_grade` | 0.6164 | 0.2692 | — |
| HistGradientBoosting | 0.6279 | 0.2680 | +0.0115 |
| XGBoost | 0.6037 | 0.2494 | **−0.0127** |

Hai ensemble nằm **dưới incumbent**. Tôi báo cáo điều đó thay vì tune cho đến khi nó biến
mất. Việc mô hình dễ diễn giải thắng là kết quả trung thực ở đây, và cũng là kết quả có thể
đem ra trình bày trước cơ quan quản lý.

---

## Phase 11 — Calibration: xác suất **chính là** sản phẩm

ROC-AUC thuần tuý là thống kê xếp hạng. Một mô hình có thể xếp hạng hoàn hảo mà vẫn sai hệ
thống về **mức** — và trong rủi ro tín dụng, mức mới là thứ được dùng: tổn thất kỳ vọng là
`PD × EAD × LGD`, và việc định giá tiêu thụ chính con số đó.

Tệ hơn, tôi đã **tự tạo ra** vấn đề này: `class_weight="balanced"` và `scale_pos_weight` bóp
méo tiên nghiệm của lớp để giúp mô hình học, nên đầu ra thô **phóng đại** xác suất default.

**Quyết định:** isotonic calibration, fit trên một lát giữ riêng của dữ liệu **train** — cụ
thể là **những tháng muộn nhất trước mốc cắt**, không phải một mẫu ngẫu nhiên, để trật tự
thời gian được giữ xuyên suốt: fit trên giai đoạn sớm, calibrate trên giai đoạn muộn hơn,
test trên giai đoạn sau nữa.

Đường calibration xác nhận: các điểm chưa calibrate nằm **dưới** đường chéo rõ rệt (dự đoán
≫ thực tế); sau isotonic chúng nằm trên đường chéo, và Brier score cải thiện từ **0.2315
xuống 0.1477**. ROC-AUC **không đổi** — calibration là phép đơn điệu nên không thể thay đổi
thứ hạng. Chính tính bất biến đó là lý do AUC một mình là không đủ.

---

## Phase 12 — Ngưỡng là quyết định chính sách, không phải đầu ra của mô hình

Ở mức prevalence 20%, con số 0.5 là vô nghĩa. Một mô hình có thể calibrate tốt mà vẫn đặt
gần như toàn bộ hồ sơ dưới 0.5 — phân loại tất cả là "sẽ trả được", đạt 80% accuracy, và vô
dụng. (Tôi **chứng minh bằng số** ở notebook 01 thay vì khẳng định suông — đó cũng là lý do
accuracy xuất hiện trong bảng của tôi **chỉ để bị bác bỏ**.)

Ba quy tắc ứng viên — tối ưu F1, Youden's J, và dựa trên chi phí. Chỉ quy tắc thứ ba mã hoá
đúng bài toán thật: một ca default bị bỏ sót mất phần gốc chưa thu hồi; một khách tốt bị từ
chối mất phần biên lãi. Với `FN:FP = 4:1`, ngưỡng rơi vào khoảng **0.195**, thấp hơn 0.5 rất
nhiều, và mô hình **đúng đắn** đánh đổi precision để lấy recall.

**Chọn trên lát calibration và áp nguyên vẹn sang test.** Tinh chỉnh ngưỡng dựa trên điểm
của tập test là cùng một họ leakage với việc fit scaler trên nó.

---

## Phase 13 — Đánh giá: con số tổng hợp là phần ít thú vị nhất

**Các metric tôi báo cáo và vì sao** — PR-AUC là chính (lớp hiếm cũng là lớp đắt, và sàn của
nó là prevalence chứ không phải 0.5); Brier đồng-chính (metric duy nhất trừng phạt
miscalibration); ROC-AUC, KS và Gini là bộ ba quy ước của ngành tín dụng; precision /
recall / F1 tại điểm vận hành đã được biện minh; accuracy chỉ như một vật trưng bày cảnh
báo.

**Phân tích lỗi.** AUC tổng hợp che giấu mọi thứ mang tính vận hành. Tôi cắt lát hiệu năng
theo grade, mục đích vay, kỳ hạn, dải FICO và — quan trọng nhất — **quý giải ngân**. Lát cắt
theo thời gian đặt ra câu hỏi mà con số tổng hợp không trả lời được: *mô hình còn chạy tốt ở
cuối cửa sổ test, hay chỉ tốt ở đầu?* FNR tăng dần theo từng quý là dạng drift đắt đỏ. Chính
độ dốc đó ấn định nhịp retrain ở Phase 15 — **suy ra chứ không phỏng đoán**.

**Công bằng (fairness).** Quyết định cho vay ảnh hưởng tới con người, nên tôi đo tỷ lệ từ
chối, FPR và FNR theo vùng của Mỹ, nhóm thu nhập và tình trạng nhà ở, kèm tỷ số disparate
impact so với ngưỡng quy ước 0.8.

Cách diễn giải mới là phần quan trọng: một nhóm có tỷ lệ từ chối cao hơn **và** tỷ lệ default
thực tế cao hơn tương ứng thì đang được đối xử nhất quán. Một nhóm có tỷ lệ từ chối cao hơn
**nhưng tỷ lệ default tương đương và FPR cao hơn** thì đang bị chính mô hình trừng phạt, chứ
không phải bị rủi ro của bản thân họ. Chỉ nhìn tỷ lệ từ chối thì không phân biệt được hai
trường hợp.

**Cảnh báo tôi luôn gắn kèm:** dataset này **không chứa thuộc tính được bảo vệ** nào. Vùng,
thu nhập, nhà ở đều là proxy. Một chênh lệch tìm thấy ở đây là có thật và đáng điều tra;
nhưng **không có** chênh lệch ở đây **không** chứng nhận mô hình công bằng trên những thuộc
tính thực sự có ý nghĩa pháp lý. Đây là một phép sàng, không phải compliance audit — và
không project nào trong số đã khảo sát làm được đến mức này.

**Phép đo mức lạc quan ảo.** Tôi chạy lại toàn bộ một lần với random split. Mọi mô hình đều
đạt điểm cao hơn. Khoảng chênh đó chính là độ lớn của ảo tưởng trong các bảng xếp hạng công
bố cho dataset này, và là lý do giao thức split chung không phải thủ tục hành chính.

---

## Phase 14 — Giải thích mô hình, và một kết quả tôi không tin

Permutation importance (mô hình dựa vào cái gì), hệ số logistic (hướng và độ mạnh), SHAP
(quy kết cho từng hồ sơ — đúng câu hỏi mà thông báo từ chối tín dụng phải trả lời theo luật
ở nhiều nơi), partial dependence (hình dạng của từng hiệu ứng và nó có đơn điệu không).

`int_rate` và `sub_grade` chiếm ưu thế. Cách đọc của tôi trong notebook là mô hình đang
**học lại phần lớn chính hệ thống thẩm định của Lending Club** chứ không thêm thông tin mới.

**Nhưng tôi phải nói rõ rằng khẳng định này chưa được kiểm chứng, và phép thử công bố duy
nhất lại chỉ theo hướng ngược lại.** Một project được khảo sát đã bỏ hẳn `grade`/`sub_grade`
và AUC chỉ dịch từ 0.7350 → 0.7335 — về cơ bản là không đổi (`docs/related_work.md`). Diễn
giải của tôi là hợp lý nhưng **tôi chưa xứng đáng với nó**. Phép ablation này rẻ và đã nằm
trong danh sách việc cần làm.

Nguyên tắc chung tôi giữ ở đây: partial dependence và SHAP mô tả **mô hình**, không mô tả
thế giới. Nếu một hướng mâu thuẫn với kiến thức thẩm định, hãy nghi leakage hoặc lỗi dữ liệu
trước khi tuyên bố một khám phá.

---

## Phase 15 — Đưa ra thực tế, và biết khi nào nó đã chết

**Đóng gói.** Tôi lưu **toàn bộ `Pipeline`** — tiền xử lý, mô hình, calibration — thành một
artifact joblib duy nhất. Chỉ lưu mỗi estimator là lỗi triển khai kinh điển: production khi
đó phải tự cài lại phần tiền xử lý, và bất kỳ sai lệch nào cũng âm thầm làm hỏng mọi dự đoán.

Nó được fit lại trên toàn bộ giai đoạn train (bao gồm cả lát calibration — hợp lệ, vì lát đó
vốn luôn là dữ liệu train). **Giai đoạn test vẫn không bao giờ bị chạm tới**, và đó chính là
thứ giữ cho các metric của nó là ước lượng trung thực.

Tôi khẳng định một phép **round-trip**: nạp lại từ đĩa và xác nhận dự đoán **giống hệt từng
bit**. Ghi được một file không có nghĩa là ghi được một file **dùng được**, và đây là phép
kiểm bắt lỗi transformer không pickle được hay lệch phiên bản **ở đây** thay vì ở production.

Một model card ghi lại cửa sổ huấn luyện, danh sách feature, các metric, hash của split và
phiên bản thư viện.

**Giám sát.** Hai kiểu hỏng khác nhau cần hai dụng cụ khác nhau:

- **Data drift** — đầu vào dịch chuyển. Đo bằng **PSI** cho từng feature so với baseline
  train (0.1 điều tra / 0.25 retrain). Quan trọng nhất: **PSI của điểm số không cần nhãn**,
  nên có ngay trong ngày hồ sơ được chấm, trong khi kết quả thật phải chờ hàng năm. Đây là
  cảnh báo sớm nhất tồn tại.
- **Concept drift** — đầu vào trông y hệt nhưng quan hệ của chúng với default đã đổi. PSI
  mù trước chuyện này. Chỉ kết quả thực tế mới phát hiện được, và đó là lý do có đường cong
  suy giảm hiệu năng theo quý.

Chính sách retrain gắn các ngưỡng số với hành động cụ thể, kèm một phân biệt vận hành đáng
nêu: **recalibration không phải retraining**. Nếu xếp hạng vẫn đúng mà chỉ xác suất lệch,
thì fit lại riêng lớp isotonic rẻ hơn và ít rủi ro hơn nhiều.

---

## Phase 16 — Những gì vẫn còn sai ở đây

Phần trung thực. Mỗi mục đều nêu đích danh công trình làm tốt hơn.

**1. Kết quả là synthetic.** Các con số phía trên đến từ một mẫu sinh ra thay thế, vì file
Kaggle 1.6GB không có trong môi trường. Chúng chỉ smoke-test đường ống và **không phải phát
hiện**. Mọi thứ khác ở đây đều bị chặn bởi việc chạy trên file thật.

**2. Tôi ghi chú right-censoring; người khác đã giải quyết nó.** Một project được khảo sát
giới hạn vào các khoản vay 36 tháng giải ngân 2012–2015, toàn bộ đã đáo hạn tính đến bản
chụp 2018 Q4 — chỉ còn 0.025% chưa ngã ngũ. Đó là thiết kế **tốt hơn hẳn** cách tôi làm là
lọc rồi viết limitation. Đây là thay đổi có giá trị cao nhất hiện có.

**3. Tỷ lệ chi phí của tôi là bịa.** `4:1` là giá trị tạm. Một project được khảo sát đã
**suy ra** biên lãi thực từ dữ liệu và thấy ngưỡng tối ưu **tăng gấp đôi** (0.25 → 0.50) so
với giá trị tạm của họ. Mọi con số phụ thuộc ngưỡng mà tôi báo cáo đều dựa trên một phỏng
đoán mà bằng chứng cho thấy là có trọng lượng thật.

**4. Không có khoảng tin cậy cho mức lift.** Tôi báo +0.0257 so với B1 dưới dạng ước lượng
điểm. Công trình đã công bố báo cáo khoảng tin cậy paired-bootstrap. Không có nó, tôi không
thể khẳng định mức lift khác 0 một cách có ý nghĩa thống kê.

**5. Luật chống leakage được ghi chép, không được cưỡng chế.** Của tôi là một danh sách viết
ra cộng với audit thủ công. Hai project được khảo sát **làm fail hẳn lượt huấn luyện** một
cách tự động nếu một cột bị cấm xuất hiện. Cách của họ sống sót qua một lần chỉnh sửa bất
cẩn trong tương lai; cách của tôi phụ thuộc vào việc người sau có đọc tài liệu hay không.

**6. `issue_d` là proxy cho ngày nộp đơn.** Nó là ngày **giải ngân**; dataset không bao giờ
ghi lại thời điểm ra quyết định. Mọi độ trễ từ nộp đơn đến giải ngân đều vô hình, nên các
feature của tôi được gán thời điểm muộn hơn một chút so với những gì một scorecard thật nhìn
thấy. Đây là thực hành chuẩn trên dataset này, nhưng phải **nói ra**, không được mặc định.

**7. Chỉ có hồ sơ được duyệt.** Mô hình ước lượng rủi ro **có điều kiện trên việc được
duyệt**, không phải cho toàn bộ dân số nộp đơn. Đây là bài toán reject inference và không
thể khắc phục bằng cách join file hồ sơ bị từ chối, vì file đó về bản chất không có kết quả.

---

## Nhật ký quyết định, gói trong một bảng

| # | Quyết định | Phương án bị loại | Vì sao |
|---|---|---|---|
| 1 | Chốt baseline trước khi mô hình hoá | So các mô hình với nhau | Lift so với incumbent mới là khẳng định có nghĩa |
| 2 | Whitelist `usecols` + drop phòng vệ | Bỏ cột leakage sau khi load | Hai lớp; lỗi này âm thầm và chí mạng |
| 3 | Chỉ giữ trạng thái cuối | Coi `Current` là đã trả xong | Sẽ gán nhãn "thành công" cho khoản vay chưa ngã ngũ |
| 4 | Split out-of-time | Random stratified | Triển khai thật là chấm điểm hồ sơ tương lai |
| 5 | Mốc ngày, không phải seed | `random_state=42` | Tái lập được qua mọi cách cài đặt |
| 6 | Giữ tỷ lệ ~50/50 | Dịch mốc cắt để đạt 80/20 | Chọn split theo kết quả nó tạo ra |
| 7 | Cắt theo biên miền cố định | Phân vị 1%/99% | Phân vị là thứ học từ dữ liệu |
| 8 | Cột chỉ báo thiếu | `fillna(median)` | Ô trống là thông tin mang tính bảo vệ |
| 9 | Sàng cột chỉ trên train | Sàng trên toàn frame | Tỷ lệ thiếu là một thống kê |
| 10 | Mã hoá grade theo thứ tự | One-hot | Vứt bỏ thứ tự đơn điệu |
| 11 | Không dùng target encoding | WOE / mean encoding trên `addr_state` | Mỗi dòng sẽ nhìn thấy đáp án của chính nó |
| 12 | Mọi thứ có học đều trong `Pipeline` | Biến đổi rồi mới split | Làm nhiễm bẩn thành bất khả thi về cấu trúc |
| 13 | `TimeSeriesSplit` cho CV | `StratifiedKFold` | Fold xáo trộn khôi phục lại look-ahead |
| 14 | Isotonic calibration trên tháng train muộn | Không calibrate / lát ngẫu nhiên | Xác suất chính là sản phẩm |
| 15 | Ngưỡng dựa trên chi phí | 0.5 | 0.5 vô nghĩa ở prevalence 20% |
| 16 | Báo cáo việc ensemble thua B1 | Tune đến khi chúng thắng | Đó **là** kết quả |
| 17 | Lưu toàn bộ pipeline | Chỉ lưu estimator | Production không được cài lại tiền xử lý |
| 18 | PSI điểm số làm giám sát chính | Chờ kết quả thực tế | Nhãn đến sau nhiều năm |

---

## Nếu bạn chỉ đọc một điều

Những quyết định ảnh hưởng nhiều nhất tới con số cuối cùng **không phải là thuật toán**.
Theo thứ tự mức ảnh hưởng: split có tôn trọng thời gian không, tiền xử lý có tôn trọng split
không, ngưỡng quyết định đặt ở đâu, và mô hình có được so với chính quy tắc mà nó định thay
thế hay không. Lựa chọn mô hình xếp thứ năm, cách xa phía sau — và **mô hình đơn giản nhất
đã thắng**.

**Tham chiếu:** so sánh benchmark kèm trích dẫn ở `docs/related_work.md`; quy tắc và quy ước
ở `CLAUDE.md`; phần cài đặt ở `notebooks/01`–`04`.
