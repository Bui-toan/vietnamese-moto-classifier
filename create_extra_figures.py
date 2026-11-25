import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Diagram 1: Software System Architecture
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('#FFFFFF')

# Tang 1: Presentation Layer
rect1 = patches.FancyBboxPatch((0.5, 4.3), 10, 1.3, boxstyle='round,pad=0.1,rounding_size=0.15',
                              facecolor='#EFF6FF', edgecolor='#1D4ED8', linewidth=2)
ax.add_patch(rect1)
ax.text(1.0, 5.2, 'TẦNG GIAO DIỆN NGƯỜI DÙNG (Streamlit Presentation Layer)', fontsize=11, fontweight='bold', color='#1E40AF')
ax.text(5.5, 4.7, 'Tab 1: Nhận diện 1 xe / đa xe   |   Tab 2: So sánh 3 mô hình   |   Tab 3: Bảng nghiệm thu   |   Tab 4: Thử độ bền Robustness', 
        fontsize=9, ha='center', color='#1E3A8A')

# Tang 2: Pipeline Layer
rect2 = patches.FancyBboxPatch((0.5, 2.4), 10, 1.4, boxstyle='round,pad=0.1,rounding_size=0.15',
                              facecolor='#F0FDF4', edgecolor='#15803D', linewidth=2)
ax.add_patch(rect2)
ax.text(1.0, 3.4, 'TẦNG ĐIỀU PHỐI VÀ TIỀN XỬ LÝ (Pipeline Controller & Logic Layer)', fontsize=11, fontweight='bold', color='#166534')
ax.text(5.5, 2.9, 'Auto-Detection (YOLOv8 nano)  ->  Crop Bounding Box  ->  Letterbox Resize (160x160)  ->  Chuẩn hóa RGB [0, 1]\nTemperature Scaling (T=0.65)  ->  Bộ lọc an toàn (Safety Reject Filter)  ->  Bộ phân tích Grad-CAM', 
        fontsize=8.8, ha='center', color='#14532D')

# Tang 3: Engine / Core AI Layer
rect3 = patches.FancyBboxPatch((0.5, 0.4), 10, 1.5, boxstyle='round,pad=0.1,rounding_size=0.15',
                              facecolor='#FAF5FF', edgecolor='#7E22CE', linewidth=2)
ax.add_patch(rect3)
ax.text(1.0, 1.5, 'TẦNG LÕI MÔ HÌNH HỌC MÁY & HỌC SÂU (AI Inference Engine)', fontsize=11, fontweight='bold', color='#6B21A8')
ax.text(5.5, 0.9, 'M1: HOG (1.764-d) + LinearSVM đa lớp (NumPy thuần)\nM2: Baseline CNN-1 (5 Conv Blocks, PyTorch)\nM3: ResNet-14 + SE-Attention Block + Focal Loss (Ensemble 2 Checkpoints M3A, M3B)', 
        fontsize=8.8, ha='center', color='#581C87')

# Inter-layer arrows
ax.annotate('', xy=(5.5, 4.3), xytext=(5.5, 3.8), arrowprops=dict(arrowstyle='<->', lw=2.2, color='#0284C7'))
ax.annotate('', xy=(5.5, 2.4), xytext=(5.5, 1.9), arrowprops=dict(arrowstyle='<->', lw=2.2, color='#0284C7'))

ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')
plt.title('Sơ đồ kiến trúc phân tầng hệ thống phần mềm nhận diện xe máy', fontsize=12, fontweight='bold', pad=15, color='#0F172A')
plt.tight_layout()
plt.savefig('reports/figures/software_system_architecture.png', bbox_inches='tight')
plt.close()

# Diagram 2: Rejection Flow
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('#FFFFFF')

r1 = patches.FancyBboxPatch((3.0, 4.9), 3.0, 0.8, boxstyle='round,pad=0.1', facecolor='#E2E8F0', edgecolor='#334155', lw=1.5)
ax.add_patch(r1)
ax.text(4.5, 5.3, 'Patch ảnh xe (160×160)', ha='center', va='center', fontweight='bold', color='#1E293B')

r2 = patches.FancyBboxPatch((2.2, 3.6), 4.6, 0.8, boxstyle='round,pad=0.1', facecolor='#DBEAFE', edgecolor='#1D4ED8', lw=1.5)
ax.add_patch(r2)
ax.text(4.5, 4.0, 'ResNet-14 + SE-Attention Suy luận\nSoftmax với Temperature Scaling (T=0.65)', ha='center', va='center', fontweight='bold', color='#1E40AF', fontsize=9)

r3 = patches.Polygon([[4.5, 2.8], [6.5, 2.2], [4.5, 1.6], [2.5, 2.2]], facecolor='#FEF3C7', edgecolor='#D97706', lw=1.8)
ax.add_patch(r3)
ax.text(4.5, 2.2, 'Max Conf >= 50% ?\nvà\nTop1 - Top2 >= 15% ?', ha='center', va='center', fontweight='bold', color='#B45309', fontsize=8.5)

r4_pass = patches.FancyBboxPatch((6.0, 0.4), 2.7, 0.8, boxstyle='round,pad=0.1', facecolor='#DCFCE7', edgecolor='#15803D', lw=1.8)
ax.add_patch(r4_pass)
ax.text(7.35, 0.8, 'Chấp nhận kết quả\n(Gán nhãn xe & vẽ Grad-CAM)', ha='center', va='center', fontweight='bold', color='#166534', fontsize=8.5)

r4_fail = patches.FancyBboxPatch((0.3, 0.4), 2.7, 0.8, boxstyle='round,pad=0.1', facecolor='#FEE2E2', edgecolor='#DC2626', lw=1.8)
ax.add_patch(r4_fail)
ax.text(1.65, 0.8, 'Kích hoạt từ chối an toàn\n("Không đủ thông tin nhận diện")', ha='center', va='center', fontweight='bold', color='#991B1B', fontsize=8.5)

# Arrows
ax.annotate('', xy=(4.5, 4.4), xytext=(4.5, 4.9), arrowprops=dict(arrowstyle='->', lw=2, color='#334155'))
ax.annotate('', xy=(4.5, 2.8), xytext=(4.5, 3.6), arrowprops=dict(arrowstyle='->', lw=2, color='#334155'))

# Branching
ax.annotate('', xy=(7.35, 1.2), xytext=(6.5, 2.2), arrowprops=dict(arrowstyle='->', lw=2, color='#15803D', connectionstyle='arc3,rad=-0.1'))
ax.text(7.1, 1.9, 'ĐẠT (Tin cậy)', color='#15803D', fontweight='bold', fontsize=9)

ax.annotate('', xy=(1.65, 1.2), xytext=(2.5, 2.2), arrowprops=dict(arrowstyle='->', lw=2, color='#DC2626', connectionstyle='arc3,rad=0.1'))
ax.text(1.0, 1.9, 'KHÔNG ĐẠT (Lạ / Che khuất)', color='#DC2626', fontweight='bold', fontsize=9)

ax.set_xlim(0, 9)
ax.set_ylim(0, 6)
ax.axis('off')
plt.title('Lưu đồ thuật toán cơ chế an toàn từ chối nhận diện (Safety Threshold Rejection)', fontsize=11, fontweight='bold', pad=15, color='#0F172A')
plt.tight_layout()
plt.savefig('reports/figures/rejection_mechanism_flow.png', bbox_inches='tight')
plt.close()

print('Successfully generated software_system_architecture.png and rejection_mechanism_flow.png!')
