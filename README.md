# Dự đoán vỡ nợ Lending Club (nhánh `simple`)

Dự đoán một khoản vay có bị **xóa nợ (Charged Off)** hay không, chỉ từ thông tin biết được lúc xét duyệt.
Project môn *Học thống kê* (DS66A / DS66B). Nhánh `simple` là bản làm lại **đơn giản, theo từng bước**:
mỗi hàm và mỗi kết quả nằm ở một ô code riêng, và mọi con số/nhận xét đều do code in ra từ dữ liệu thật.

> Bản cũ (4 notebook, nhiều kỹ thuật hơn) nằm trong `archive/`.

## Dữ liệu

Kaggle: [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
Đặt `accepted_2007_to_2018Q4.csv.gz` vào `data/raw/` (file bị git bỏ qua). Không có dữ liệu giả lập:
thiếu file thì notebook dừng và báo rõ.

## Cách chạy

```bash
pip install -r requirements.txt
jupyter lab        # chạy notebook 1 rồi notebook 2, mỗi cái từ trên xuống dưới
```

Notebook 1 mất vài phút (nạp file nén lớn, đỉnh khoảng 10 GB RAM). Notebook 2 mất khoảng 10 phút.

## Các bước đã làm

| Notebook | Bước | Nội dung |
|---|---|---|
| `01_hieu_du_lieu_va_chia_tap` | 1 | Tóm tắt đề bài; từ điển giải thích đủ 151 cột (có `assert` khớp header) |
| | 2 | Nạp **toàn bộ dữ liệu gốc**, đối chiếu từ điển với dữ liệu thật, định nghĩa nhãn bằng bằng chứng |
| | 3 | EDA về cấu trúc thời gian; luật "đã đáo hạn"; chia **70 / 15 / 15 theo thời gian** (kiểu Arturo-GA) |
| `02_lam_sach_va_feature` | 4 | Làm sạch; pipeline học từ train, áp sang val/test; ô chứng minh không rò rỉ |
| | 5 | Gán 151 cột vào nhóm (mỗi nhóm có bằng chứng), feature engineering, lọc tương quan, pipeline hoàn chỉnh |

Các bước mô hình hóa và đánh giá sẽ thêm sau.

## Kết quả của bước 1-5 (đọc từ output notebook)

**Chia dữ liệu** (notebook 1):

| | Số dòng | Giai đoạn phát hành | Tỷ lệ vỡ nợ |
|---|---|---|---|
| train | 464.185 | 2010-01 đến 2015-04 | 0,148 |
| val | 95.405 | 2015-05 đến 2015-08 | 0,148 |
| test | 107.434 | 2015-09 đến 2015-12 | 0,148 |

- File gốc 2.260.701 dòng, 151 cột; 1.345.310 dòng có nhãn; sau khi lọc từ 2010 và đã đáo hạn còn 667.024 dòng.
- Luật đáo hạn kiểm chứng bằng code: nhóm đã đáo hạn có 99,6% khoản vay đã có kết quả, nhóm chưa đáo hạn 42,4%.

**Chọn feature** (notebook 2): 151 cột gốc -> 57 cột ứng viên -> 46 feature số + 6 feature phân loại,
tức 161 cột sau mã hóa one-hot. Cột bị loại theo nhóm: sau khi vay / không rõ thời điểm (42), trống hoàn toàn
trong train (32), (gần như) hằng số (7), thiếu quá nhiều trong train (5), quá nhiều giá trị chữ (4), định danh (2),
nhãn và cột phụ (2).

## Hạn chế đã biết (đều do notebook đo được)

1. **Val và test chỉ có khoản vay 36 tháng** (100%), còn train có 11,8% khoản 60 tháng, vì khoản 60 tháng phát hành
   muộn chưa đủ thời gian đáo hạn. Test chỉ gồm 4 tháng cuối 2015.
2. **32 cột trống hoàn toàn trong train**, trong đó 17 cột chỉ bắt đầu có dữ liệu ở giai đoạn val/test, nên không dùng được.
3. **34 cột bureau chỉ có dữ liệu từ 2012** và bị điền trung vị cho các khoản vay 2010-2011.
4. **5 cột dạng "số tháng kể từ sự kiện"** bị loại vì thiếu trên 30% trong train, dù giá trị thiếu có thể có nghĩa là
   "chưa từng xảy ra"; chưa kiểm tra bằng code xem việc đó có mất tín hiệu không.
5. `issue_d` (ngày giải ngân) được dùng thay cho ngày nộp đơn; chỉ có khoản vay đã được chấp nhận trong dữ liệu.
6. Cách chia này **thay cho mốc chung 2016-01-01** của bản cũ; các nhóm khác dùng cùng dataset cần đối chiếu
   `md5_id_train` trong `artifacts/split_manifest.json`.

## Ghi chú về repo

- `CLAUDE.md`: quy tắc làm việc cho các phiên Claude Code (nguyên tắc "mọi kết luận có code suy ra").
- `docs/`: tài liệu của bản cũ, có số liệu từ nguồn ngoài và các bước không còn trong notebook mới.
  Coi là lỗi thời cho đến khi được viết lại.
