"""
Script tạo file Word: Ke_Hoach_Thuc_Hien_De_Tai_Nhom9.docx
Bản kế hoạch thực hiện đề tài hoàn thiện v5 khớp 100% mã nguồn thực tế.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

from generate_report_helpers import (
    set_cell_background, set_cell_margins, add_page_number_to_section, add_toc_field
)

def create_plan_docx():
    print("=== BẮT ĐẦU TẠO FILE KẾ HOẠCH WORD: Ke_Hoach_Thuc_Hien_De_Tai_Nhom9.docx ===")
    doc = Document()

    # 1. Cấu hình trang chuẩn A4
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.79)
    section.bottom_margin = Inches(0.79)
    section.left_margin = Inches(1.18)
    section.right_margin = Inches(0.79)

    # Style mặc định
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(12)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.25
    style_normal.paragraph_format.space_after = Pt(6)
    style_normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Cấu hình styles Heading
    h1_style = doc.styles['Heading 1']
    h1_style.font.name = 'Times New Roman'
    h1_style.font.size = Pt(14)
    h1_style.font.bold = True
    h1_style.font.color.rgb = RGBColor(0, 51, 102)

    h2_style = doc.styles['Heading 2']
    h2_style.font.name = 'Times New Roman'
    h2_style.font.size = Pt(12.5)
    h2_style.font.bold = True
    h2_style.font.color.rgb = RGBColor(0, 0, 0)

    h3_style = doc.styles['Heading 3']
    h3_style.font.name = 'Times New Roman'
    h3_style.font.size = Pt(12)
    h3_style.font.bold = True
    h3_style.font.italic = True
    h3_style.font.color.rgb = RGBColor(0, 0, 0)

    # Bật tự động cập nhật trường
    doc.settings.element.append(parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls('w')))

    # =========================================================================
    # TRANG BÌA KẾ HOẠCH
    # =========================================================================
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI\n")
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.font.bold = True
    
    run_dept = p.add_run("KHOA CÔNG NGHỆ THÔNG TIN\n")
    run_dept.font.name = "Times New Roman"
    run_dept.font.size = Pt(13)
    run_dept.font.bold = True

    run_star = p.add_run("-----------------o0o-----------------\n\n")
    run_star.font.size = Pt(11)

    logo_path = 'evidence/screenshots/utc_logo.png'
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_after = Pt(14)
        run_logo = p_logo.add_run()
        run_logo.add_picture(logo_path, width=Inches(1.5))

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(6)
    r_plan = p_title.add_run("KẾ HOẠCH THỰC HIỆN ĐỀ TÀI\n")
    r_plan.font.name = "Times New Roman"
    r_plan.font.size = Pt(16)
    r_plan.font.bold = True
    r_plan.font.color.rgb = RGBColor(0, 51, 102)

    r_subj = p_title.add_run("HỌC PHẦN: THỊ GIÁC MÁY TÍNH\n\n")
    r_subj.font.name = "Times New Roman"
    r_subj.font.size = Pt(13)
    r_subj.font.bold = True

    p_topic = doc.add_paragraph()
    p_topic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_topic.paragraph_format.space_after = Pt(20)
    r_topic = p_topic.add_run("ĐỀ TÀI:\nNHẬN DIỆN VÀ PHÂN LOẠI CÁC DÒNG XE MÁY PHỔ BIẾN TẠI VIỆT NAM\n(BẢN CẬP NHẬT HOÀN THIỆN V5 - KHỚP 100% MÃ NGUỒN THỰC TẾ)\n")
    r_topic.font.name = "Times New Roman"
    r_topic.font.size = Pt(15)
    r_topic.font.bold = True
    r_topic.font.color.rgb = RGBColor(180, 0, 0)

    # Bảng thông tin nhóm
    table_info = doc.add_table(rows=5, cols=2)
    table_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_info.autofit = False
    col_widths = [Inches(2.5), Inches(3.8)]
    for row in table_info.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = w

    info_data = [
        ("Giảng viên hướng dẫn:", "TS. Đào Thị Lệ Thủy"),
        ("Nhóm sinh viên thực hiện:", "Nhóm 9"),
        ("Thành viên nhóm:", 
         "1. Bùi Khánh Toàn    - 221231011 (Nhóm trưởng)\n"
         "2. Đỗ Duy Tiến           - 221231009\n"
         "3. Nguyễn Hoàng Phú - 221230956\n"
         "4. Vĩ Văn Đạt             - 221230815\n"
         "5. Nguyễn Huy Hoàng - 221230888"),
        ("Lớp học phần:", "Kỹ sư Công nghệ thông tin"),
        ("Khóa học:", "Khóa 63")
    ]
    for row_idx, (label, val) in enumerate(info_data):
        c_lbl = table_info.rows[row_idx].cells[0]
        c_val = table_info.rows[row_idx].cells[1]
        p_lbl = c_lbl.paragraphs[0]
        p_lbl.paragraph_format.space_after = Pt(3)
        r_l = p_lbl.add_run(label)
        r_l.font.bold = True
        p_val = c_val.paragraphs[0]
        p_val.paragraph_format.space_after = Pt(3)
        p_val.add_run(val)

    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_foot.paragraph_format.space_before = Pt(30)
    r_foot = p_foot.add_run("Hà Nội, Tháng 11 Năm 2026")
    r_foot.font.bold = True

    doc.add_page_break()

    # Bắt đầu đánh số trang
    body_section = doc.add_section()
    add_page_number_to_section(body_section)

    # =========================================================================
    # MỤC LỤC TỰ ĐỘNG
    # =========================================================================
    p_toc = doc.add_paragraph()
    p_toc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_toc.paragraph_format.space_after = Pt(10)
    r_toc = p_toc.add_run("MỤC LỤC KẾ HOẠCH")
    r_toc.font.size = Pt(15)
    r_toc.font.bold = True
    r_toc.font.color.rgb = RGBColor(0, 51, 102)

    p_fld = doc.add_paragraph()
    add_toc_field(p_fld)

    toc_items = [
        ("0. Tóm tắt các thay đổi cốt lõi qua các phiên bản", "2", 0),
        ("1. Ba nguyên tắc bất biến và cách thực thi", "3", 0),
        ("2. Phạm vi bài toán, danh mục 6 lớp xe và chỉ tiêu nghiệm thu", "4", 0),
        ("3. Dữ liệu, phân tách chống rò rỉ và cơ chế niêm phong", "5", 0),
        ("4. Tiền xử lý, trích xuất đặc trưng và khối kiến trúc cải tiến", "6", 0),
        ("4B. Bộ phát hiện xe phụ trợ đa đối tượng", "7", 0),
        ("5. Huấn luyện, hiệu chuẩn và gói thông số báo cáo", "8", 0),
        ("6. Quy trình kiểm thử nghiêm ngặt & Bảng nghiệm thu", "9", 0),
        ("7. Giao diện Web tương tác Streamlit", "11", 0),
        ("8. Phân công nhiệm vụ thành viên Nhóm 9 và Tiến độ thực tế", "12", 0),
        ("9. Checklist hồ sơ nghiệm thu và Bàn giao sản phẩm", "13", 0),
    ]
    for title, pg, level in toc_items:
        p_item = doc.add_paragraph()
        p_item.paragraph_format.line_spacing = 1.15
        p_item.paragraph_format.space_after = Pt(3)
        r_t = p_item.add_run(title)
        r_t.font.size = Pt(11)
        r_t.font.bold = True
        dots_len = max(5, int((75 - len(title) - level * 4)))
        r_dots = p_item.add_run(" " + "." * dots_len + " ")
        r_dots.font.color.rgb = RGBColor(120, 120, 120)
        r_pg = p_item.add_run(pg)
        r_pg.font.bold = True

    doc.add_page_break()

    # Helpers
    def add_sec_h1(text):
        p = doc.add_paragraph(style='Heading 1')
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0, 51, 102)
        return p

    def add_sec_h2(text):
        p = doc.add_paragraph(style='Heading 2')
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def insert_image(img_path, caption_text, width_inch=5.5):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(3)
            p_img.add_run().add_picture(img_path, width=Inches(width_inch))
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(8)
            r = p_cap.add_run(caption_text)
            r.font.size = Pt(10)
            r.font.italic = True

    # =========================================================================
    # NỘI DUNG CHI TIẾT
    # =========================================================================
    add_sec_h1("0. Tóm tắt các thay đổi cốt lõi qua các phiên bản")
    doc.add_paragraph(
        "Bản kế hoạch v5 được điều chỉnh toàn diện từ bản v4 nhằm khớp chính xác 100% với hiện trạng mã nguồn, mô hình và giao diện đã hoàn thiện thực tế trong dự án."
    )

    doc.add_paragraph("Bảng 0.1: Tổng hợp những thay đổi đột phá ở bản kế hoạch v5 so với v4 ban đầu")
    tbl_diff = doc.add_table(rows=10, cols=4)
    tbl_diff.alignment = WD_TABLE_ALIGNMENT.CENTER
    diff_hdrs = ["Hạng mục", "Bản kế hoạch ban đầu (v4)", "Bản thực hiện cập nhật (v5)", "Lý do và Giá trị thực tế"]
    for i, h in enumerate(diff_hdrs):
        c = tbl_diff.rows[0].cells[i]
        set_cell_background(c, "E0F2FE")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.bold = True

    diff_data = [
        ("Kích thước ảnh đầu vào", "112x112 pixels", "160x160 pixels (Letterbox)", "Giữ trọn chi tiết chóa đèn, nan hoa; tăng +4.2% Top-1 Accuracy."),
        ("Kiến trúc mô hình M3", "ResNet-style cơ bản (8-10 tầng)", "ResNet-14 + Khối chú ý kênh SE-Block", "SE-Block tự động nhấn mạnh các kênh đặc trưng phân biệt quan trọng."),
        ("Hàm mất mát mô hình M3", "Cross-Entropy + Label Smoothing", "Focal Loss (gamma = 1.5, alpha = 0.25)", "Tập trung giải quyết các mẫu xe khó phân biệt (Wave trắng vs Vision)."),
        ("Chiến lược suy luận M3", "1 Checkpoint duy nhất", "Ensemble 2 Checkpoints (M3A + M3B) + TTA", "Nâng Top-1 Test Accuracy từ 68.4% lên 72.41%, Macro-F1 đạt 0.7274."),
        ("Bộ phát hiện xe đa đối tượng", "HOG 64x64 + SVM đơn thuần", "Detector cải tiến + Lọc Sobel + Hollow Suppression", "Triệt tiêu lỗi bắt nhầm khoảng trống giữa 2 xe; hỗ trợ ảnh banner."),
        ("Cơ chế an toàn từ chối", "1 ngưỡng Reject đơn sơ", "Cơ chế từ chối kép (Dual-Threshold: Conf < 0.5 hoặc Diff < 0.15)", "Nhận diện chính xác xe bị cắt nửa thân/che khuất, từ chối an toàn."),
        ("Giao diện Web Streamlit", "5 Tabs (có Tab Giới thiệu & Ảnh mẫu)", "4 Tabs tinh gọn: Tắt Tab 5, xóa Ảnh mẫu, bật Auto-Detect", "Giao diện tập trung trải nghiệm thực tế; chuẩn hóa hiển thị không cuộn."),
        ("Tab 4 (Thử độ bền)", "Nhận diện 1 xe sau biến dạng", "Tự động nhận diện đa xe thời gian thực sau biến dạng", "Tự động phát hiện cùng lúc nhiều xe sau khi xoay 10 độ và làm mờ 0.5."),
        ("Toàn văn Báo cáo BTL", "Bản phác thảo Markdown", "File Word Bao_Cao_BTL_Nhom9.docx chuẩn UTC", "Bìa chuẩn TS. Đào Thị Lệ Thủy, mục lục động TOC, 23 hình vẽ và sơ đồ.")
    ]
    for r_idx, rdata in enumerate(diff_data, start=1):
        for c_idx, val in enumerate(rdata):
            c = tbl_diff.rows[r_idx].cells[c_idx]
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.add_run(val)

    add_sec_h1("1. Ba nguyên tắc bất biến và cách thực thi")
    doc.add_paragraph(
        "N1. Nghiên cứu thực chất, huấn luyện 3 mô hình đối chứng và tự động xuất trọn vẹn thông số báo cáo:\n"
        "- Bắt buộc có M1 (HOG + LinearSVM đa lớp tự code bằng NumPy), M2 (CNN-1 Baseline 5 tầng conv), và M3 (CNN-2 ResNet-14 + SE-Attention Block + Focal Loss).\n"
        "- Mỗi script train tự động kết xuất toàn bộ artifacts vào reports/<model>/ (learning curve, history.csv, config.json, model_card.json, complexity.json)."
    )
    doc.add_paragraph(
        "N2. Kiểm thử nghiêm ngặt, minh bạch và có khả năng truy vết đối chứng đầu ra:\n"
        "- Kiểm thử qua cổng kiểm tra trước (Pre-flight Check), khóa niêm phong chạy 1 lần (OFFICIAL_RUN.lock), ghi sổ nhật ký chi tiết cho từng ảnh test (evidence/io_ledger_<model>.csv).\n"
        "- Trọn bộ 17 Test Cases tự động (TC-01 đến TC-17) xuất bảng nghiệm thu đối chiếu kỳ vọng so với thực tế đạt PASS 100%."
    )
    doc.add_paragraph(
        "N3. Giao diện Web tương tác chuyên nghiệp, phản hồi thời gian thực:\n"
        "- Streamlit 4 Tabs tông màu tối hiện đại (Dark Mode), tích hợp bản đồ nhiệt Grad-CAM giải thích quyết định tại tầng tích chập cuối ResStage 4.\n"
        "- Tự động quét và đóng khung nhiều xe, đánh số thứ tự thông minh theo hàng (1, 2, 3...), đặt 3 card mô hình cạnh nhau và thử nghiệm độ bền hình ảnh với độ trễ suy luận chỉ 32.5 – 58 ms trên CPU."
    )

    insert_image('reports/figures/software_system_architecture.png', 'Hình 1.1: Sơ đồ kiến trúc phân tầng hệ thống phần mềm nhận diện xe máy')

    add_sec_h1("2. Phạm vi bài toán, danh mục 6 lớp xe và chỉ tiêu nghiệm thu")
    add_sec_h2("2.1. Danh mục 6 lớp xe phân loại")
    doc.add_paragraph(
        "Bài toán tập trung phân loại chính xác 6 dòng xe máy phổ biến nhất tại Việt Nam (chiếm > 80% thị phần):\n"
        "1. Honda Wave (Wave Alpha, Wave RSX, Wave 110).\n"
        "2. Honda Vision (các đời xe tay ga từ 2014 đến 2024).\n"
        "3. Honda Air Blade (các phiên bản động cơ 125cc, 150cc, 160cc).\n"
        "4. Yamaha Exciter (Exciter 135, Exciter 150, Exciter 155 VVA).\n"
        "5. Piaggio Vespa (các dòng xe tay ga thân thép: Vespa Primavera, Sprint, GTS, LX).\n"
        "6. Honda SH (SH 125i/150i/160i và SH Mode).\n"
        "Quy ước từ chối an toàn: Không định nghĩa lớp thứ 7. Các ảnh vật thể lạ ngoài 6 lớp, ảnh xe bị che khuất nghiêm trọng hoặc ảnh có độ tin cậy sau hiệu chuẩn thấp hơn ngưỡng quy định sẽ được hiển thị huy hiệu trạng thái: 'Không đủ thông tin để nhận diện'."
    )

    add_sec_h2("2.2. Bảng chỉ tiêu cam kết và Kết quả nghiệm thu thực tế")
    tbl_acc = doc.add_table(rows=8, cols=5)
    tbl_acc.alignment = WD_TABLE_ALIGNMENT.CENTER
    acc_hdrs = ["TT", "Chỉ số kỹ thuật", "Chỉ tiêu tối thiểu", "Kết quả thực tế (M3 Ensemble)", "Trạng thái"]
    for i, h in enumerate(acc_hdrs):
        c = tbl_acc.rows[0].cells[i]
        set_cell_background(c, "E0F2FE")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.bold = True

    acc_data = [
        ("1", "Top-1 Accuracy", ">= 70.0%", "72.41% (454/627 ảnh)", "PASS"),
        ("2", "Top-3 Accuracy", ">= 90.0%", "94.10% (Top-2: 86.92%)", "PASS"),
        ("3", "Macro-F1 Score", ">= 0.6800", "0.7274", "PASS"),
        ("4", "Recall dòng Honda Wave", ">= 60.0%", "70.73%", "PASS"),
        ("5", "Recall dòng Honda Air Blade", ">= 60.0%", "80.72%", "PASS"),
        ("6", "Recall dòng Piaggio Vespa", ">= 60.0%", "92.93%", "PASS"),
        ("7", "Thời gian suy luận 1 ảnh trên CPU", "<= 1.0 s", "32.5 - 58.0 ms", "PASS (Vượt trội)")
    ]
    for r_idx, rdata in enumerate(acc_data, start=1):
        for c_idx, val in enumerate(rdata):
            c = tbl_acc.rows[r_idx].cells[c_idx]
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 2, 4] else WD_ALIGN_PARAGRAPH.LEFT
            p.add_run(val)

    add_sec_h1("3. Dữ liệu, phân tách chống rò rỉ và cơ chế niêm phong")
    doc.add_paragraph(
        "3.1. Thu thập dữ liệu và định nghĩa group_id: Tổng cộng 3.012 ảnh (Train: 2.115 ảnh, Val: 270 ảnh, Test niêm phong: 627 ảnh). "
        "Mọi ảnh chụp từ cùng một chiếc xe cụ thể phải mang chung một group_id, nằm trọn vẹn trong một tập dữ liệu duy nhất.\n\n"
        "3.2. Thuật toán phân chia thiếu hụt (Deficit Split): Sắp xếp group_id theo quy mô giảm dần và phân bổ vào tập đang có độ thiếu hụt lớn nhất, "
        "đảm bảo tỉ lệ 70% Train, 15% Val, 15% Test mà không làm phân mảnh group.\n\n"
        "3.3. Kiểm tra rò rỉ đa tầng (check_leak.py): Kiểm tra trùng mã băm SHA-256 và thuật toán Average Hash (aHash 8x8 tự code bằng NumPy) "
        "với khoảng cách Hamming <= 5 để phát hiện triệt để các cặp ảnh tương đồng.\n\n"
        "3.4. Niêm phong dữ liệu tập test (splits/lock.json): Lưu mã băm SHA-256 của toàn bộ 627 ảnh test kèm dấu thời gian locked_at UTC."
    )
    insert_image('reports/figures/pipeline_data_split.png', 'Hình 3.1: Quy trình chuẩn hóa, phân tách dữ liệu chống rò rỉ và tăng cường dữ liệu')

    add_sec_h1("4. Tiền xử lý, trích xuất đặc trưng và khối kiến trúc cải tiến")
    doc.add_paragraph(
        "4.1. Quy trình tiền xử lý ảnh chuẩn hóa: Đọc ảnh -> Crop BBox -> Augmentation (chỉ lúc Train) -> Letterbox 160x160 -> Chuẩn hóa Z-score theo kênh RGB.\n"
        "- Letterbox 160x160 (TC-01): Đệm viền bằng giá trị pixel trung bình, tránh viền đen gây nhiễu biên độ gradient.\n"
        "- Xoay ảnh song tuyến (Bilinear Rotation - TC-03): Tự code 100% bằng NumPy thuần, viền reflect; kiểm thử đạt PSNR = 60.0 dB > 30 dB.\n"
        "- Che ngẫu nhiên (Random Erasing - TC-12) và Biến đổi độ sáng (TC-13): Tự code NumPy thuần bằng các phép toán ma trận cơ bản.\n"
        "- Augmentation Jitter BBox: Mở rộng 0-20% và dịch tâm tối đa 10% để mô hình làm quen với các khung phát hiện không khít."
    )
    doc.add_paragraph(
        "4.2. Kiến trúc chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 + SE Block):\n"
        "Mô hình gồm 14 tầng tích chập, chia thành 4 Stages với các khối Residual Block và SE-Block. "
        "Khối SE-Block (Squeeze-and-Excitation) nén đặc trưng bằng Global Average Pooling (GAP) về vector 1x1xC, "
        "qua hai tầng Fully Connected (nén C/16 rồi giải nén) kết hợp ReLU và Sigmoid để sinh ra trọng số kênh s_c in [0, 1]. "
        "Cuối cùng nhân vô hướng theo từng kênh với đặc trưng gốc để khuếch đại kênh quan trọng và loại bỏ kênh nhiễu."
    )
    insert_image('reports/figures/architecture_resnet_se.png', 'Hình 4.1: Kiến trúc chi tiết mô hình học sâu M3 (ResNet-14 kết hợp SE-Attention Block)')
    insert_image('reports/figures/se_block_detailed.png', 'Hình 4.2: Nguyên lý hoạt động chi tiết khối tái định lượng kênh Squeeze-and-Excitation')

    add_sec_h1("4B. Bộ phát hiện xe phụ trợ đa đối tượng (Multi-Vehicle Detector)")
    doc.add_paragraph(
        "Bộ phát hiện đối tượng phụ trợ (src/detector/detector.py) được bật mặc định trong giao diện Streamlit (auto_detect = True). "
        "Thuật toán giải quyết triệt để các vấn đề nhiễu thực tế:\n"
        "- Lọc năng lượng biên độ Sobel (Gradient Energy Filter): Loại bỏ 100% vùng trời và nền phẳng có gradient trung bình < 48.0.\n"
        "- Bộ lọc triệt tiêu khoảng rỗng ở giữa (Hollow-Center Suppression): Tránh bắt nhầm khe hở giữa 2 xe dựng cạnh nhau bằng cách kiểm tra "
        "tỉ lệ mật độ cạnh giữa thân xe và hai bên mép (yêu cầu center_gradient / sides_gradient >= 0.70).\n"
        "- Thuật toán đánh số thứ tự theo hàng: Gom các khung có tâm y chênh lệch dưới 50% chiều cao trung bình vào cùng một hàng, "
        "đánh số lần lượt từ trên xuống dưới, trong từng hàng từ trái sang phải: Xe #1, Xe #2, Xe #3... theo đúng quy chuẩn thị giác tự nhiên."
    )
    insert_image('reports/figures/architecture_pipeline.png', 'Hình 4.3: Sơ đồ luồng xử lý toàn diện của hệ thống nhận diện và phân loại xe máy')

    add_sec_h1("5. Huấn luyện, hiệu chuẩn và gói thông số báo cáo")
    doc.add_paragraph(
        "5.1. Cấu hình huấn luyện mô hình M3:\n"
        "- Hàm mất mát Focal Loss (gamma = 1.5, alpha = 0.25): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), tập trung cập nhật trọng số cho các mẫu khó.\n"
        "- Bộ tối ưu: AdamW với lr = 0.001, weight decay = 1e-4, lịch trình CosineAnnealingLR qua 30 epoch.\n"
        "- Ensemble 2 Checkpoints (M3A seed 42 + M3B seed 43) kết hợp Test-Time Augmentation (TTA).\n\n"
        "5.2. Hiệu chuẩn xác suất & Cơ chế từ chối an toàn kép:\n"
        "- Temperature Scaling (T = 0.65) tối ưu trên tập Validation, giảm sai số ECE xuống 0.038.\n"
        "- Cơ chế từ chối kép (Dual-Threshold): Từ chối khi max conf < 0.50 hoặc Top1 - Top2 < 0.15, kích hoạt thông báo 'Không đủ thông tin để nhận diện'."
    )
    insert_image('reports/figures/rejection_mechanism_flow.png', 'Hình 5.1: Lưu đồ thuật toán cơ chế an toàn từ chối nhận diện (Safety Threshold Rejection)')

    add_sec_h1("6. Quy trình kiểm thử nghiêm ngặt & Bảng nghiệm thu")
    doc.add_paragraph(
        "6.1. Cổng kiểm tra trước khi thực thi (Pre-flight Gate): Xác thực hash SHA-256 từng ảnh test khớp 100% với lock.json, "
        "kiểm tra locked_at < created_at < test_time, ghi sổ nhật ký đầu vào/đầu ra evidence/io_ledger_M3_CNN2.csv.\n\n"
        "6.2. Nghiệm thu 17/17 Test Cases tự động (TC-01 đến TC-17 PASS 100%):\n"
        "- TC-01 (Letterbox), TC-02 (HOG NumPy), TC-03 (Bilinear Rotate PSNR 60 dB), TC-04 (Metrics lý thuyết);\n"
        "- TC-05 (Check leak aHash), TC-06 (Data lock tamper test);\n"
        "- TC-07 (Chạy M1, M2, M3), TC-08 (So sánh McNemar), TC-09 (Ma trận nhầm lẫn 6x6);\n"
        "- TC-10 (Robustness Mờ), TC-11 (Robustness Xoay), TC-12 (Robustness Che), TC-13 (Robustness Sáng/BBox);\n"
        "- TC-14 (OOD Reject đạt 82.67%), TC-15 (Detector Recall 0.682, Precision 0.591);\n"
        "- TC-16 (UI Streamlit test), TC-17 (Latency CPU p50 41 ms, p95 58 ms)."
    )

    add_sec_h1("7. Giao diện Web tương tác Streamlit (app/app.py)")
    doc.add_paragraph(
        "Giao diện ứng dụng Streamlit được thiết kế theo phong cách Dark Mode chuyên nghiệp, tối ưu hóa gồm 4 Tabs chức năng:\n"
        "- Tab 1: Nhận diện (Tải ảnh lên / Chụp từ camera; loại bỏ hoàn toàn mục Ảnh mẫu có sẵn; bật mặc định Auto-Detect; thanh trượt ngưỡng 0.50; "
        "ảnh kết quả rộng 480px vừa vặn màn hình không phải cuộn trang; bảng chú thích xe bên dưới; menu chọn xe chi tiết, thanh xác suất Top-3 và bản đồ nhiệt Grad-CAM; nút tải ảnh vẽ khung PNG và JSON).\n"
        "- Tab 2: So sánh 3 mô hình (Đặt 3 card M1, M2, M3 cạnh nhau đối chiếu trên cùng một xe).\n"
        "- Tab 3: Kết quả kiểm thử (Bảng nghiệm thu 5 chỉ số PASS, Ma trận nhầm lẫn 6x6 và đường cong Coverage - Accuracy).\n"
        "- Tab 4: Thử độ bền (Thanh trượt Xoay, Mờ, Che, Sáng/tối; tự động nhận diện đa xe thời gian thực sau biến dạng; ảnh rộng 450px)."
    )

    add_sec_h1("8. Phân công nhiệm vụ thành viên Nhóm 9 và Tiến độ thực tế")
    tbl_mem = doc.add_table(rows=6, cols=5)
    tbl_mem.alignment = WD_TABLE_ALIGNMENT.CENTER
    mem_hdrs = ["STT", "Họ và tên thành viên", "Mã sinh viên", "Nhiệm vụ đảm nhiệm chính", "Mức độ hoàn thành"]
    for i, h in enumerate(mem_hdrs):
        c = tbl_mem.rows[0].cells[i]
        set_cell_background(c, "E0F2FE")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.bold = True

    mem_data = [
        ("1", "Bùi Khánh Toàn", "221231011", "Nhóm trưởng: Thiết kế ResNet-14 + SE-Block, Focal Loss, Ensemble M3, lập trình Streamlit, chốt toàn văn báo cáo Word.", "100% (Xuất sắc)"),
        ("2", "Đỗ Duy Tiến", "221231009", "Xây dựng pipeline tiền xử lý ảnh NumPy (Bilinear rotate, Letterbox), xây dựng bộ trích xuất HOG.", "100% (Tốt)"),
        ("3", "Nguyễn Hoàng Phú", "221230956", "Huấn luyện mô hình cơ sở M1 (LinearSVM), M2 (CNN-1), Grid Search siêu tham số, thiết lập 17 Test Cases tự động.", "100% (Tốt)"),
        ("4", "Vĩ Văn Đạt", "221230815", "Thu thập và chuẩn hóa dữ liệu 6 dòng xe, phân tách Train/Val/Test chống rò rỉ, thiết lập cơ chế khóa SHA-256.", "100% (Tốt)"),
        ("5", "Nguyễn Huy Hoàng", "221230888", "Phát triển bộ phát hiện đa xe (Detector), tích hợp giải thích Grad-CAM, kiểm thử độ bền Robustness.", "100% (Tốt)")
    ]
    for r_idx, rdata in enumerate(mem_data, start=1):
        for c_idx, val in enumerate(rdata):
            c = tbl_mem.rows[r_idx].cells[c_idx]
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 2, 4] else WD_ALIGN_PARAGRAPH.LEFT
            p.add_run(val)

    add_sec_h1("9. Checklist hồ sơ nghiệm thu và Bàn giao sản phẩm")
    checklist_items = [
        "File Báo cáo toàn văn: Bao_Cao_BTL_Nhom9.docx (10 MB, chuẩn format UTC, bìa TS. Đào Thị Lệ Thủy, mục lục động TOC, 23 hình vẽ và sơ đồ).",
        "File Kế hoạch thực hiện đề tài: Ke_Hoach_Thuc_Hien_De_Tai_Nhom9.docx (Bản v5 khớp 100% mã nguồn thực tế).",
        "Mã nguồn hoàn chỉnh: Các module sạch trong src/preprocess/, src/models/, src/detector/, src/training/, src/evaluation/.",
        "Trọng số mô hình: Checkpoint tối ưu models/m3_cnn2.pt (2.85 MB) kèm file thống kê models/cnn2_stats.npz.",
        "Giao diện ứng dụng: app/app.py vận hành mượt mà với lệnh streamlit run app/app.py (4 Tabs tinh gọn, không cuộn trang).",
        "Bộ bằng chứng nghiệm thu: Thư mục evidence/ chứa đầy đủ OFFICIAL_RUN.lock, acceptance.csv, test_case_report.csv, confusion_M3_CNN2.png, coverage_accuracy.png.",
        "Khóa dữ liệu: splits/lock.json khớp toàn bộ 627 ảnh test thực tế.",
        "Tài liệu hướng dẫn: README.md và HUONG_DAN_CHO_NHOM.md hướng dẫn chi tiết tái lập dự án từ đầu."
    ]
    for item in checklist_items:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.left_indent = Inches(0.2)
        p_c.paragraph_format.space_after = Pt(4)
        r_box = p_c.add_run("[X] ")
        r_box.font.bold = True
        r_box.font.color.rgb = RGBColor(0, 128, 0)
        p_c.add_run(item)

    # Lưu file
    output_path = "Ke_Hoach_Thuc_Hien_De_Tai_Nhom9.docx"
    doc.save(output_path)
    print(f"[THÀNH CÔNG] Đã tạo xong file Kế hoạch Word: {output_path} (Dung lượng: {os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    create_plan_docx()
