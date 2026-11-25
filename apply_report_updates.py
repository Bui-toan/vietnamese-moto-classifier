import re

with open('build_full_report_docx.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update style_normal and add heading styles + updateFields
old_style_block = """    style_normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY"""

new_style_block = """    style_normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Cấu hình styles cho Heading để Word tự động liên kết Mục lục TOC và Navigation Pane
    h1_style = doc.styles['Heading 1']
    h1_style.font.name = 'Times New Roman'
    h1_style.font.size = Pt(15)
    h1_style.font.bold = True
    h1_style.font.color.rgb = RGBColor(0, 51, 102)

    h2_style = doc.styles['Heading 2']
    h2_style.font.name = 'Times New Roman'
    h2_style.font.size = Pt(13)
    h2_style.font.bold = True
    h2_style.font.color.rgb = RGBColor(0, 0, 0)

    h3_style = doc.styles['Heading 3']
    h3_style.font.name = 'Times New Roman'
    h3_style.font.size = Pt(13)
    h3_style.font.bold = True
    h3_style.font.italic = True
    h3_style.font.color.rgb = RGBColor(0, 0, 0)

    # Bật cờ tự động cập nhật trường khi mở file trong Microsoft Word
    doc.settings.element.append(parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls('w')))"""

content = content.replace(old_style_block, new_style_block, 1)

# 2. Update add_chapter_title, add_section_title, add_subsection_title to use styles
old_fn_block = """    def add_chapter_title(title_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(10)
        run = p.add_run(title_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 51, 102)
        return p

    def add_section_title(sec_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(sec_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_subsection_title(subsec_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(subsec_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.bold = True
        return p"""

new_fn_block = """    def add_chapter_title(title_text):
        p = doc.add_paragraph(style='Heading 1')
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(10)
        run = p.add_run(title_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 51, 102)
        return p

    def add_section_title(sec_text):
        p = doc.add_paragraph(style='Heading 2')
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(sec_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_subsection_title(subsec_text):
        p = doc.add_paragraph(style='Heading 3')
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(subsec_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.bold = True
        return p"""

content = content.replace(old_fn_block, new_fn_block, 1)

# 3. Update DANH MỤC HÌNH VẼ
old_figs_list = """    figs_list = [
        ("Hình 3.1", "Quy trình chuẩn hóa, phân tách dữ liệu chống rò rỉ và tăng cường dữ liệu", "16"),
        ("Hình 4.1", "Sơ đồ luồng xử lý toàn diện của hệ thống nhận diện và phân loại xe máy", "19"),
        ("Hình 4.2", "Kiến trúc chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 kết hợp SE-Attention)", "21"),
        ("Hình 5.1", "Đường cong huấn luyện (Loss và Accuracy) của mô hình M3 qua 30 epoch", "26"),
        ("Hình 5.2", "Ma trận nhầm lẫn chuẩn hóa 6x6 của mô hình M3 Ensemble trên 627 ảnh test", "27"),
        ("Hình 5.3", "Lưới các trường hợp nhận diện đúng điển hình kèm bản đồ nhiệt Grad-CAM", "28"),
        ("Hình 5.4", "Lưới các trường hợp dự đoán sai điển hình kèm bản đồ nhiệt Grad-CAM", "29"),
        ("Hình 5.5", "Giao diện nhận diện 1 xe độc lập (Honda Vision đạt 74,0%)", "30"),
        ("Hình 5.6", "Chi tiết nhận diện 1 xe và bản đồ nhiệt Grad-CAM tập trung vào thân xe", "30"),
        ("Hình 5.7", "Giao diện tự động phát hiện và khoanh vùng đa xe trên ảnh banner phức tạp", "31"),
        ("Hình 5.8", "Chi tiết Xe #1 (Honda Vision đạt 82,82%) và bản đồ nhiệt Grad-CAM", "31"),
        ("Hình 5.9", "Cơ chế an toàn từ chối nhận diện vùng nghi vấn Xe #2 (độ tin cậy 46,45% < 50%)", "32"),
        ("Hình 5.10", "Chi tiết Xe #3 (Honda Air Blade đạt 81,52%) và bản đồ nhiệt Grad-CAM", "32"),
        ("Hình 5.11", "Giao diện Tab đối chiếu kết quả trực tiếp giữa 3 mô hình M1, M2 và M3", "33"),
        ("Hình 5.12", "Bảng nghiệm thu 5 chỉ số kiểm thử chính thức trên giao diện Streamlit", "33"),
        ("Hình 5.13", "Ma trận nhầm lẫn và đường cong Coverage - Accuracy trên giao diện Web", "34"),
        ("Hình 5.14", "Giao diện Tab thử nghiệm độ bền thời gian thực (Xoay 10 độ, làm mờ 0,5)", "34"),
        ("Hình 5.15", "Phân tích trường hợp giới hạn khi xe bị cắt nửa thân và nhầm lẫn màu sắc", "35")
    ]"""

new_figs_list = """    figs_list = [
        ("Hình 3.1", "Quy trình chuẩn hóa, phân tách dữ liệu chống rò rỉ và tăng cường dữ liệu", "16"),
        ("Hình 4.1", "Sơ đồ kiến trúc phân tầng hệ thống phần mềm nhận diện xe máy", "19"),
        ("Hình 4.2", "Sơ đồ luồng xử lý toàn diện của hệ thống nhận diện và phân loại xe máy", "20"),
        ("Hình 4.3", "Kiến trúc chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 kết hợp SE-Attention)", "21"),
        ("Hình 4.4", "Nguyên lý hoạt động chi tiết khối tái định lượng kênh Squeeze-and-Excitation", "22"),
        ("Hình 4.5", "Lưu đồ thuật toán cơ chế an toàn từ chối nhận diện (Safety Threshold Rejection)", "23"),
        ("Hình 5.1", "Đường cong huấn luyện (Loss và Accuracy) của mô hình M3 qua 30 epoch", "26"),
        ("Hình 5.2", "Ma trận nhầm lẫn chuẩn hóa 6x6 của mô hình M3 Ensemble trên 627 ảnh test", "27"),
        ("Hình 5.3", "Lưới các trường hợp nhận diện đúng điển hình kèm bản đồ nhiệt Grad-CAM", "28"),
        ("Hình 5.4", "Lưới các trường hợp dự đoán sai điển hình kèm bản đồ nhiệt Grad-CAM", "29"),
        ("Hình 5.5", "Giao diện nhận diện 1 xe độc lập (Honda Vision đạt 74,0%)", "30"),
        ("Hình 5.6", "Chi tiết nhận diện 1 xe và bản đồ nhiệt Grad-CAM tập trung vào thân xe", "30"),
        ("Hình 5.7", "Giao diện tự động phát hiện và khoanh vùng đa xe trên ảnh banner phức tạp", "31"),
        ("Hình 5.8", "Chi tiết Xe #1 (Honda Vision đạt 82,82%) và bản đồ nhiệt Grad-CAM", "31"),
        ("Hình 5.9", "Cơ chế an toàn từ chối nhận diện vùng nghi vấn Xe #2 (độ tin cậy 46,45% < 50%)", "32"),
        ("Hình 5.10", "Chi tiết Xe #3 (Honda Air Blade đạt 81,52%) và bản đồ nhiệt Grad-CAM", "32"),
        ("Hình 5.11", "Giao diện Tab đối chiếu kết quả trực tiếp giữa 3 mô hình M1, M2 và M3", "33"),
        ("Hình 5.12", "Bảng nghiệm thu 5 chỉ số kiểm thử chính thức trên giao diện Streamlit", "33"),
        ("Hình 5.13", "Ma trận nhầm lẫn và đường cong Coverage - Accuracy trên giao diện Web", "34"),
        ("Hình 5.14", "Giao diện Tab thử nghiệm độ bền thời gian thực (Xoay 10 độ, làm mờ 0,5)", "34"),
        ("Hình 5.15", "Phân tích trường hợp nhầm lẫn thực tế: Wave đỏ nhận đúng, Wave trắng nhầm sang Vision", "35"),
        ("Hình 5.16", "Phân tích trường hợp xe bị che khuất và cắt nửa thân kích hoạt cảnh báo an toàn", "35")
    ]"""

content = content.replace(old_figs_list, new_figs_list, 1)

# 4. Update Chapter 4 with all new design diagrams
old_ch4 = """    # =========================================================================
    # CHƯƠNG 4: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG
    # =========================================================================
    add_chapter_title("CHƯƠNG 4: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG")

    add_section_title("4.1. Sơ đồ kiến trúc tổng thể của hệ thống")
    doc.add_paragraph(
        "Hệ thống nhận diện và phân loại xe máy được thiết kế theo kiến trúc module hóa hướng đối tượng, bao gồm 4 khối chức năng chính: "
        "(1) Khối tiếp nhận và tiền xử lý hình ảnh; (2) Khối phát hiện và định vị khung xe (Detector); "
        "(3) Khối suy luận học sâu phân loại dòng xe (Classifier M3); và (4) Khối tương tác người dùng, giải thích quyết định và cảnh báo an toàn."
    )

    insert_image_with_caption('reports/figures/architecture_pipeline.png', 'Hình 4.1: Sơ đồ luồng xử lý toàn diện của hệ thống nhận diện và phân loại xe máy', 6.0)

    add_section_title("4.2. Thiết kế chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 + SE Block)")
    doc.add_paragraph(
        "Mô hình học sâu chủ đạo M3 được nhóm thiết kế tối ưu trong file src/models/cnn2.py với 14 tầng tích chập, chia thành 4 giai đoạn (Stages). "
        "Mỗi giai đoạn gồm 2 khối Residual Block có kết nối tắt Skip Connection và tích hợp cơ chế chú ý kênh Squeeze-and-Excitation. "
        "Toàn bộ mô hình có kích thước nhỏ gọn (chỉ 2,85 MB với xấp xỉ 2,8 triệu tham số), cực kỳ tối ưu cho việc suy luận thời gian thực trên CPU thông thường."
    )

    insert_image_with_caption('reports/figures/architecture_resnet_se.png', 'Hình 4.2: Kiến trúc chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 kết hợp SE-Attention)', 6.0)"""

new_ch4 = """    # =========================================================================
    # CHƯƠNG 4: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG
    # =========================================================================
    add_chapter_title("CHƯƠNG 4: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG")

    add_section_title("4.1. Sơ đồ kiến trúc tổng thể của hệ thống")
    doc.add_paragraph(
        "Hệ thống nhận diện và phân loại xe máy được thiết kế theo kiến trúc phân tầng module hóa hướng đối tượng (Hình 4.1), bao gồm 3 tầng chức năng độc lập: "
        "(1) Tầng giao diện người dùng (Streamlit Presentation Layer) trực quan, phản hồi thời gian thực; "
        "(2) Tầng điều phối và tiền xử lý (Pipeline Controller & Logic Layer) chịu trách nhiệm tự động phát hiện khung xe, chuẩn hóa Letterbox và hiệu chuẩn xác suất; "
        "(3) Tầng lõi học máy và học sâu (AI Inference Engine) chứa bộ phân loại chủ lực M3 (ResNet-14 + SE-Block) cùng các mô hình đối chứng M1 và M2."
    )
    insert_image_with_caption('reports/figures/software_system_architecture.png', 'Hình 4.1: Sơ đồ kiến trúc phân tầng hệ thống phần mềm nhận diện xe máy', 6.0)

    doc.add_paragraph(
        "Luồng xử lý dữ liệu đầu cuối (End-to-End Pipeline) kết nối chặt chẽ các module (Hình 4.2): khi ảnh được tải lên, bộ phát hiện phụ trợ quét và trích xuất từng xe máy, "
        "chuẩn hóa kích thước qua thuật toán Letterbox giữ nguyên tỉ lệ gốc, sau đó đưa qua mạng ResNet-14 để suy luận và giải thích bằng Grad-CAM."
    )
    insert_image_with_caption('reports/figures/architecture_pipeline.png', 'Hình 4.2: Sơ đồ luồng xử lý toàn diện của hệ thống nhận diện và phân loại xe máy', 6.0)

    add_section_title("4.2. Thiết kế chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 + SE Block)")
    doc.add_paragraph(
        "Mô hình học sâu chủ đạo M3 được nhóm thiết kế tối ưu trong file src/models/cnn2.py với 14 tầng tích chập, chia thành 4 giai đoạn (Stages). "
        "Mỗi giai đoạn gồm 2 khối Residual Block có kết nối tắt Skip Connection và tích hợp cơ chế chú ý kênh Squeeze-and-Excitation. "
        "Toàn bộ mô hình có kích thước nhỏ gọn (chỉ 2,85 MB với xấp xỉ 2,8 triệu tham số), cực kỳ tối ưu cho việc suy luận thời gian thực trên CPU thông thường (Hình 4.3)."
    )
    insert_image_with_caption('reports/figures/architecture_resnet_se.png', 'Hình 4.3: Kiến trúc chi tiết mô hình học sâu M3 (CNN-2 ResNet-14 kết hợp SE-Attention)', 6.0)

    doc.add_paragraph(
        "Cơ chế chú ý kênh Squeeze-and-Excitation (SE Block) đóng vai trò then chốt giúp mô hình tự động học được tầm quan trọng tương đối của từng kênh đặc trưng (Hình 4.4). "
        "Giai đoạn Squeeze sử dụng Global Average Pooling để nén không gian H × W thành vector 1 × 1 × C. "
        "Giai đoạn Excitation sử dụng hai tầng Fully Connected (nén theo tỉ lệ r = 16 rồi giải nén) kết hợp hàm kích hoạt ReLU và Sigmoid để sinh ra trọng số tỉ lệ s_c trong đoạn [0, 1]. "
        "Cuối cùng, đặc trưng gốc được nhân vô hướng theo từng kênh với s_c (Scale), giúp mạng khuếch đại các kênh chứa đặc trưng phân biệt (như pô xe Wave, đèn Vision) và triệt tiêu kênh nhiễu."
    )
    insert_image_with_caption('reports/figures/se_block_detailed.png', 'Hình 4.4: Nguyên lý hoạt động chi tiết khối tái định lượng kênh Squeeze-and-Excitation (SE Block)', 6.0)"""

content = content.replace(old_ch4, new_ch4, 1)

# In Chapter 4.3, add the Rejection flowchart
old_sec_43 = """    add_section_title("4.3. Thiết kế bộ phát hiện xe phụ trợ đa đối tượng")
    doc.add_paragraph(
        "Bộ phát hiện đối tượng phụ trợ (src/detector/detector.py) thực hiện quét ảnh đa tỉ lệ và định vị các xe máy trong ảnh. "
        "Thuật toán giải quyết triệt để các vấn đề nhiễu thực tế:\\n"
        "- Lọc năng lượng biên độ Sobel (Gradient Energy Filter): Loại bỏ 100% vùng trời và nền phẳng có gradient trung bình < 48.0.\\n"
        "- Bộ lọc triệt tiêu khoảng rỗng ở giữa (Hollow-Center Suppression): Tránh bắt nhầm khe hở giữa 2 xe dựng cạnh nhau bằng cách kiểm tra "
        "tỉ lệ mật độ cạnh giữa thân xe và hai bên mép (yêu cầu center_gradient / sides_gradient >= 0.70).\\n"
        "- Thuật toán đánh số thứ tự theo hàng: Gom các khung có tâm y chênh lệch dưới 50% chiều cao trung bình vào cùng một hàng, "
        "đánh số lần lượt từ trên xuống dưới, trong từng hàng từ trái sang phải: Xe #1, Xe #2, Xe #3... theo đúng quy chuẩn thị giác tự nhiên."
    )"""

new_sec_43 = """    add_section_title("4.3. Thiết kế bộ phát hiện xe phụ trợ đa đối tượng và cơ chế từ chối an toàn")
    doc.add_paragraph(
        "Bộ phát hiện đối tượng phụ trợ (src/detector/detector.py) thực hiện quét ảnh đa tỉ lệ và định vị các xe máy trong ảnh. "
        "Thuật toán giải quyết triệt để các vấn đề nhiễu thực tế:\\n"
        "- Lọc năng lượng biên độ Sobel (Gradient Energy Filter): Loại bỏ 100% vùng trời và nền phẳng có gradient trung bình < 48.0.\\n"
        "- Bộ lọc triệt tiêu khoảng rỗng ở giữa (Hollow-Center Suppression): Tránh bắt nhầm khe hở giữa 2 xe dựng cạnh nhau bằng cách kiểm tra "
        "tỉ lệ mật độ cạnh giữa thân xe và hai bên mép (yêu cầu center_gradient / sides_gradient >= 0.70).\\n"
        "- Thuật toán đánh số thứ tự theo hàng: Gom các khung có tâm y chênh lệch dưới 50% chiều cao trung bình vào cùng một hàng, "
        "đánh số lần lượt từ trên xuống dưới, trong từng hàng từ trái sang phải: Xe #1, Xe #2, Xe #3... theo đúng quy chuẩn thị giác tự nhiên."
    )
    doc.add_paragraph(
        "Nhằm ngăn chặn các quyết định sai lầm khi gặp đối tượng lạ, vật thể không phải xe máy hoặc xe bị che khuất nghiêm trọng, "
        "hệ thống áp dụng cơ chế từ chối an toàn (Safety Threshold Rejection) với lưu đồ hoạt động như mô tả ở Hình 4.5. "
        "Nếu xác suất lớn nhất sau hiệu chuẩn Temperature Scaling (T = 0,65) nhỏ hơn ngưỡng reject_threshold (0,50), "
        "hoặc độ chênh lệch giữa Top-1 và Top-2 nhỏ hơn diff_margin (0,15), hệ thống sẽ từ chối gán nhãn và kích hoạt trạng thái cảnh báo an toàn."
    )
    insert_image_with_caption('reports/figures/rejection_mechanism_flow.png', 'Hình 4.5: Lưu đồ thuật toán cơ chế an toàn từ chối nhận diện (Safety Threshold Rejection)', 5.5)"""

content = content.replace(old_sec_43, new_sec_43, 1)

# 5. Update Chapter 5 image paths and Section 5.7
content = content.replace("'evidence/screenshots/demo_1xe_detail_vision_gradcam.jpg'", "'evidence/screenshots/demo_1xe_detail_vision_gradcam.png'")
content = content.replace("'evidence/screenshots/demo_overview_detection.jpg'", "'evidence/screenshots/demo_overview_detection.png'")
content = content.replace("'evidence/screenshots/demo_detail_xe1_vision.jpg'", "'evidence/screenshots/demo_detail_xe1_vision.png'")
content = content.replace("'evidence/screenshots/demo_detail_xe2_reject.jpg'", "'evidence/screenshots/demo_detail_xe2_reject.png'")
content = content.replace("'evidence/screenshots/demo_detail_xe3_airblade.jpg'", "'evidence/screenshots/demo_detail_xe3_airblade.png'")
content = content.replace("'evidence/screenshots/demo_tab4_robustness.jpg'", "'evidence/screenshots/demo_tab4_robustness.png'")

old_sec_57 = """    add_section_title("5.7. Phân tích các trường hợp dự đoán sai, xe bị che khuất và giới hạn mô hình")
    doc.add_paragraph(
        "Tuân thủ nghiêm ngặt yêu cầu đánh giá khách quan của giảng viên hướng dẫn, nhóm tiến hành phân tích sâu sắc các trường hợp dự đoán sai (Hình 5.15):\\n"
        "1. Sự nhầm lẫn giữa Wave RSX màu trắng và Honda Vision: Trong ảnh thử nghiệm hai chiếc xe số Wave RSX (Hình 5.15), chiếc xe màu đỏ bên trái "
        "được nhận diện chuẩn xác 100% là Honda Wave (60,6%). Tuy nhiên, chiếc xe bên phải (cùng dòng Wave RSX nhưng bản màu trắng - bạc) "
        "bị mô hình dự đoán nhầm thành Honda Vision (55,3%). Nguyên nhân là do màu sơn trắng kết hợp phông nền trắng studio làm mờ các chi tiết nan hoa, "
        "khiến cụm yếm xe và chóa đèn trông na ná đầu xe tay ga Vision. Tuy nhiên, bảng Top-3 vẫn xếp Honda Wave ở vị trí số 2 với độ tin cậy bám sát.\\n"
        "2. Trường hợp xe bị che khuất hoặc cắt nửa thân: Khi gặp ảnh ghép nhân tạo có đường phân cách trắng chia đôi khung hình, "
        "chiếc xe bên trái bị mất toàn bộ cụm bánh trước và chóa đèn. Việc mất đi các đặc trưng định danh cốt lõi khiến mô hình giảm độ tin cậy xuống 31,4% "
        "và kích hoạt cảnh báo an toàn. Những phân tích này giúp nhóm đề xuất các giải pháp nâng cấp kiến trúc mạng trong tương lai."
    )

    insert_image_with_caption('reports/test_eval/misclassified_samples_gradcam.png', 'Hình 5.15: Phân tích các trường hợp giới hạn khi xe bị cắt nửa thân và nhầm lẫn màu sắc', 5.8)"""

new_sec_57 = """    add_section_title("5.7. Phân tích các trường hợp dự đoán sai, xe bị che khuất và giới hạn mô hình")
    doc.add_paragraph(
        "Tuân thủ nghiêm ngặt yêu cầu đánh giá khách quan của giảng viên hướng dẫn, nhóm tiến hành phân tích sâu sắc các trường hợp dự đoán sai và giới hạn thực tế của mô hình:\\n"
        "1. Sự nhầm lẫn giữa xe số Wave RSX màu trắng và xe tay ga Honda Vision (Hình 5.15): Trong ảnh thử nghiệm hai chiếc xe số Honda Wave RSX, "
        "chiếc xe màu đỏ bên trái (Xe 1) được nhận diện chuẩn xác 100% là Honda Wave (60,6%). Tuy nhiên, chiếc xe bên phải (Xe 2 - cùng dòng Wave RSX nhưng có phối màu trắng - bạc) "
        "lại bị mô hình phân loại nhầm thành Honda Vision (55,3%). Khi giải mã bằng Grad-CAM, nhóm phát hiện: màu sơn trắng sáng kết hợp nền studio trắng đã triệt tiêu độ tương phản "
        "của khối động cơ và bộ nan hoa xe số. Đồng thời, đường cong mềm mại của yếm Wave RSX trắng có góc nghiêng rất tương đồng với yếm trước của dòng xe tay ga Vision. "
        "Dù vậy, phân bố xác suất Top-3 vẫn xếp Honda Wave ở vị trí số 2 rất sát sao (xấp xỉ 38%), cho thấy mô hình không bị thiên lệch hoàn toàn.\\n"
        "2. Trường hợp xe bị che khuất và cắt nửa thân trên ảnh ghép (Hình 5.16): Khi người dùng đưa vào ảnh ghép gồm nhiều xe bị đường biên trắng chia cắt ngang thân xe, "
        "chiếc xe bên trái chỉ còn lại phần đuôi và bánh sau, toàn bộ phần đầu xe, chóa đèn pha và mặt nạ nhận diện cốt lõi đều bị biến mất. "
        "Trước các mẫu vật thể khuyết thiếu này, nếu các hệ thống thông thường cố gắng đoán mò thì sẽ đưa ra nhãn sai với độ rủi ro rất cao. "
        "Trái lại, hệ thống của Nhóm 9 đã phản ứng cực kỳ chính xác theo thiết kế an toàn: độ tin cậy rơi xuống dưới 50% (đạt 46,5%), "
        "kích hoạt ngay cơ chế từ chối an toàn với thông báo 'Không đủ thông tin để nhận diện'. "
        "Đây chính là minh chứng sống động cho tính tin cậy và khả thi cao trong triển khai thực tế của đề tài."
    )
    insert_image_with_caption('evidence/screenshots/demo_fail_case_1.png', 'Hình 5.15: Phân tích trường hợp nhầm lẫn thực tế: Wave đỏ nhận đúng, Wave trắng nhầm sang Vision do nền sáng', 5.5)
    insert_image_with_caption('evidence/screenshots/demo_fail_case_2.png', 'Hình 5.16: Phân tích trường hợp xe bị che khuất và cắt nửa thân kích hoạt cơ chế an toàn từ chối nhận diện', 5.5)"""

content = content.replace(old_sec_57, new_sec_57, 1)

with open('build_full_report_docx.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully applied all updates to build_full_report_docx.py!')
