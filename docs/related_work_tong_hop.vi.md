# Related Work & Benchmarks — Bảng tổng hợp (bản tiếng Việt, đọc nội bộ)

> **Tài liệu nội bộ, không phải deliverable.** Bảng tự tổng hợp để nhìn nhanh. Bản tiếng Anh
> dùng cho báo cáo là `docs/related_work.md`. Số liệu của các repo GitHub (dòng 4–9) đã được
> đối chiếu với README gốc vào 10/2026; các dòng 1–3 chưa đối chiếu lại.

Bảng tổng hợp các công trình nghiên cứu và dự án thực nghiệm tiêu biểu liên quan đến bài toán
dự báo rủi ro vỡ nợ (Credit Default Risk) và chấm điểm tín dụng. **Nhóm A** là nền tảng
phương pháp (không dùng riêng dữ liệu Lending Club); **nhóm B** là các dự án trên chính tập
Lending Club, so sánh trực tiếp được với project này.

**Cách đọc cột Split:** **[OOT]** = chia theo thời gian · **[Ngẫu nhiên]** = chia ngẫu nhiên
· **[Không nêu]** = nguồn không nói rõ. AUC giữa các kiểu split **không so sánh trực tiếp
được** — chia ngẫu nhiên thường cao hơn OOT khoảng 1 điểm (Xia et al.).

## Nhóm A — Nền tảng phương pháp

| STT | Công trình / Dự án & Tác giả | Repository / Paper URL | Tập dữ liệu & Phạm vi lọc | Phương pháp / Mô hình chính | Cơ chế Split & Chống Data Leakage | Kết quả chính & Metric nổi bật | Đóng góp & Phát hiện đặc thù (Business / Interpretability / Fairness) |
|:---:|---|---|---|---|---|---|---|
| **1** | **Fair_Credit_Scoring**<br>*(Kozodoi, Jacob, & Lessmann)* | [kozodoi/Fair_Credit_Scoring](https://github.com/kozodoi/Fair_Credit_Scoring) | 7 bộ dữ liệu tín dụng thực tế (UCI, Kaggle, PAKDD) — **không phải Lending Club** | Đánh giá 8 phương pháp Fair ML qua 3 nhóm: Pre-, In-, Post-processors | Khảo sát đánh đổi giữa tính công bằng và lợi nhuận (profit–fairness trade-off) | Tiêu chí *Separation* phù hợp nhất cho credit scorecard | Thuật toán *In-processors* cân bằng tối ưu giữa lợi nhuận và độ công bằng; giảm kỳ thị thuật toán với tổn thất lợi nhuận thấp. |
| **2** | **SurvXGBoost**<br>*(Xia, He, Li, Fu, & Xu)* ¹ | [TEDE Journal](https://journals.vilniustech.lt/index.php/TEDE/article/view/13997) | Dữ liệu khoản vay tiêu dùng trực tuyến thực tế | SurvXGBoost (kết hợp Survival Analysis vào GBDT/XGBoost) + PCA trên các biến vĩ mô | **[OOS + OOT]** Đánh giá đồng thời out-of-sample và out-of-time | **Out-of-sample 68.07% so với out-of-time 67.07%** (AUC) — chia ngẫu nhiên cao hơn khoảng 1 điểm; SurvXGBoost vượt Cox PH, RSF về C-index, AUC, misclassification cost | Xử lý dữ liệu bị cắt (censoring) theo thời gian và tích hợp chỉ số kinh tế vĩ mô vào dự báo PD. |
| **3** | **Machine Learning in Credit Scoring**<br>*(Systematic review, 2025)* ² | [Artificial Intelligence Review (2025)](https://link.springer.com/article/10.1007/s10462-025-11416-2) | Khảo sát tổng quan hệ thống (Systematic Literature Review) | Đánh giá toàn diện các phương pháp ML trong xếp hạng tín nhiệm | Tổng kết các tiêu chuẩn thực nghiệm và phương pháp đánh giá rủi ro tín dụng | Bức tranh tổng quan về các hướng tiếp cận ML hiện đại | Khung lý thuyết nền tảng và căn cứ khoa học cho bài toán chấm điểm tín dụng. |

## Nhóm B — Dự án trên dữ liệu Lending Club

| STT | Công trình / Dự án & Tác giả | Repository / Paper URL | Tập dữ liệu & Phạm vi lọc | Phương pháp / Mô hình chính | Cơ chế Split & Chống Data Leakage | Kết quả chính & Metric nổi bật | Đóng góp & Phát hiện đặc thù (Business / Interpretability / Fairness) |
|:---:|---|---|---|---|---|---|---|
| **4** | **Predicting-Default-Clients**<br>*(Yanxia Li / yanxiali)* | [yanxiali/Predicting-Default-Clients](https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans) | LendingClub 2007–2017 (~1.6M khoản vay, 150 features) — bản dữ liệu cũ hơn file 2007–2018Q4 | Logistic Regression, Random Forest, KNN (tối ưu qua Grid Search) | **[Không nêu]** Một tập test riêng; cross-validated AUROC trên tập train chỉ dùng để chọn mô hình | Logistic Regression đạt AUROC = **0.70** trên tập test, tốc độ nhanh nhất | Dùng K-S test và tương quan Pearson lọc biến; Random Forest chỉ ra `interest rate` và `dti` quan trọng nhất. |
| **5** | **Credit Risk & Portfolio Analysis**<br>*(Arnav618)* | [Arnav618/lending-club-credit-risk](https://github.com/Arnav618/lending-club-credit-risk) | Lending Club (1.3M dòng đã kết thúc → lấy mẫu phân tầng 290k dòng) | XGBoost Classifier + `scale_pos_weight` (≈ 3.98) | **[Không theo thời gian]** 80/20 trên mẫu phân tầng. Mutual Information phát hiện leakage `total_rec_prncp` (MI = 0.47); loại >30 cột hậu giải ngân | AUC tăng từ **0.7177** lên **0.7265**; Recall tăng lên 0.66 ³ | Tính biên lãi thực tế 27.2% để chỉnh cut-off từ 0.25 lên 0.50; khung Expected Loss ($EL = PD \times LGD \times EAD$) với LGD thực tế 69.8%. |
| **6** | **Loan Default Prediction**<br>*(Nasha14)* | [Nasha14/Lending-Club-Loan-Default](https://github.com/Nasha14/Lending-Club-Loan-Default-Prediction) | Lending Club (2007–2018Q4), còn lại ~1.35M khoản vay đã kết thúc | Baseline Logistic Regression đối chứng với XGBoost | **[Không nêu]** Loại bỏ post-origination leakage (lịch sử thanh toán, đòi nợ); trích xuất độ dài lịch sử tín dụng | XGBoost: AUC = **0.735** (LogReg: 0.649); Precision = 0.33, Recall = 0.68 | **Ablation:** bỏ `grade`/`sub_grade` thì AUC chỉ giảm 0.7350 → 0.7335. **Theo tác giả**, điều này cho thấy mô hình tự tái tạo tín hiệu rủi ro độc lập với grade của sàn ⁴. |
| **7** | **credit-risk-scorecard**<br>*(Tanish-Srivastava)* | [Tanish-Srivastava/credit-risk-scorecard](https://github.com/Tanish-Srivastava/credit-risk-scorecard) | Lending Club 2007–2018 (~1.35M khoản vay đã kết thúc; bad rate ~20%) | OptBinning + WOE (ràng buộc đơn điệu) → Logistic Regression | **[Ngẫu nhiên]** Whitelist biến lúc nộp đơn; khử đa cộng tuyến ở `pub_rec_bankruptcies` & `total_acc` | AUC = **0.692**, Gini = **0.385**, KS = **0.277** (vượt LC grade baseline AUC 0.680 → lift **+0.012**) | Hiệu chuẩn tốt qua các decile (~0.5%); quy đổi điểm kiểu CIBIL (PDO = 50, base 600); chỉ ra FICO bị giảm sức mạnh do sàn đã lọc trước FICO < 627. |
| **8** | **lendingclub-default-risk**<br>*(Arturo-GA)* | [Arturo-GA/lendingclub-default-risk](https://github.com/Arturo-GA/lendingclub-default-risk) | Lending Club 2010–2018, **chỉ khoản đã đáo hạn** (kỳ hạn 36/60 tháng đã kết thúc trước 2018-12; ~800k khoản vay) | LightGBM, XGBoost, Scorecard WOE, Logistic Regression, Random Forest | **[OOT]** 70% cũ nhất train / 15% val / 15% mới nhất test; whitelist; `assert_no_leakage` làm fail lượt chạy nếu có cột cấm; giữ nguyên imbalance | LightGBM: AUC = **0.7151**, Gini = **0.4302**, KS = **0.3137**; Scorecard WOE: AUC 0.7040 | Ma trận chi phí FN:FP = 5:1 (ngưỡng 0.51, duyệt 65.8%, default giảm còn 9.07%) ⁵; SHAP và PSI đo data drift. |
| **9** | **credit-risk**<br>*(vaibhavkev)* | [vaibhavkev/credit-risk](https://github.com/vaibhavkev/credit-risk) | Lending Club 2012–2015, kỳ hạn 36 tháng, **đã đáo hạn** (589,635 khoản vay) | XGBoost với Monotone Constraints (ràng buộc đơn điệu theo logic nghiệp vụ) | **[OOT]** Train 01/2012–06/2014 / Val 07–12/2014 / Test 2015; whitelist `columns.py` + `test_leakage.py` làm fail nếu có cột cấm | XGBoost OOT AUC = **0.696**, Gini = **0.392**, KS = **0.284** (vượt LC sub-grade 0.679, lift **+0.018**, 95% CI [+0.016, +0.020]) | Quy đổi giá trị kinh doanh: từ chối 5% rủi ro nhất tăng thêm +$6.1M net cash; TreeSHAP trích 3 reason code; API serving FastAPI. |
| **★** | **Project này**<br>*(DS66A/B midterm)* | `notebooks/01`–`04` | Lending Club 2007–2018Q4, ~1.34M khoản vay đã kết thúc (lọc theo trạng thái cuối, **chưa** lọc theo đáo hạn) | Logistic Regression + isotonic calibration (mô hình đóng gói); so sánh với RF, HistGB, XGBoost | **[OOT]** Train trước 2016-01-01 / test 2016–2018; mọi biến đổi có học nằm trong `Pipeline`; `TimeSeriesSplit` cho CV; 151 → 30 cột theo luật phân loại có assertion | LR: AUC = **0.7076**, Gini = 0.415, KS = 0.301 (vượt LC sub-grade 0.6871, lift **+0.0205**, chưa có CI). XGBoost: 0.7161 (lift +0.029) | **Sàng lọc fairness** theo vùng / thu nhập / nhà ở — không dự án nào ở trên làm; phát hiện cờ đỏ ở nhóm thu nhập (DI 0.534) và nhà ở (DI 0.674). |

## Ghi chú

1. **Năm của Xia et al. cần đối chiếu.** `docs/related_work.md` ghi 2020, bảng gốc ghi 2021.
   Bài đăng online và bản in có thể khác năm; kiểm tra trang tạp chí trước khi trích dẫn.
2. **Bài tổng quan 2025 cần bổ sung tên tác giả** trước khi trích dẫn trong báo cáo.
3. **Khoảng tin cậy của Arnav618 tự mâu thuẫn trong nguồn.** README ghi bootstrap 95% CI
   [0.7127, 0.7226] cho mô hình đã tune, nhưng khoảng này không chứa chính điểm 0.7265 của
   mô hình đó (nó chứa 0.7177 của baseline). Vì vậy bảng chỉ ghi điểm số, không ghi CI.
4. **Ablation của Nasha14 chưa chứng minh được.** Nếu `int_rate` vẫn còn trong mô hình, việc
   bỏ `grade`/`sub_grade` gần như không mất thông tin, vì lãi suất được Lending Club định ra
   từ grade. Đây là nhận định của tác giả, không phải kết luận đã kiểm chứng.
5. **Ngưỡng 0.51 của Arturo-GA không so trực tiếp được với ngưỡng của project này.** Với xác
   suất đã hiệu chỉnh, ngưỡng tối ưu lý thuyết cho FN:FP = 5:1 là khoảng 1/(1+5) ≈ 0.17.
   Ngưỡng 0.51 cho thấy họ dùng điểm chưa hiệu chỉnh (có cân bằng lớp). Project này dùng
   xác suất đã hiệu chỉnh, ngưỡng khoảng 0.20 ở FN:FP = 4:1.

## Nhận xét nhanh

- **Các AUC cao nhất (0.735, 0.7265, 0.70) đều đến từ project chia ngẫu nhiên hoặc không nêu
  cách chia.** Hai project OOT (Arturo-GA 0.7151, vaibhavkev 0.696) là nhóm so sánh đúng với
  project này.
- **Cả hai project OOT đều lọc theo khoản đã đáo hạn**; project này chưa làm. Đây là điểm
  yếu phương pháp lớn nhất so với họ.
- **Chỉ hai project đo lift so với grade của Lending Club** (+0.012 và +0.018). Lift của
  project này (+0.0205 với LR) nhỉnh hơn dải đó và chưa có khoảng tin cậy.
- **Không dự án nào trong nhóm B làm fairness**; Kozodoi et al. (dòng 1) là nền tảng phương
  pháp phù hợp nếu muốn đi xa hơn phép sàng lọc proxy hiện tại.
