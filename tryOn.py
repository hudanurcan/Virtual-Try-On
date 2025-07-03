
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mediapipe as mp

class VirtualTryOn:
    def __init__(self, root):
        self.root = root
        self.root.title("Sanal Kıyafet Deneme Uygulaması")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.current_clothing = None  # Şu anki kıyafeti tutacak değişken
        self.clothing_path = ""  # Kıyafet dosyasının yolu
        self.cap = None  # Kamerayı tutacak değişken
        self.pose = None  # Mediapipe poz tespiti için değişken
        self.mp_pose = mp.solutions.pose  # Mediapipe poz çözümleme modülü
        self.mp_drawing = mp.solutions.drawing_utils  # Mediapipe için çizim araçları
        
        self.setup_ui()  # UI (Kullanıcı arayüzü) kurulum fonksiyonunu çağırıyoruz
    
    def setup_ui(self):
        # Başlık çerçevesini oluşturuyoruz
        title_frame = tk.Frame(self.root, bg="#3498db")
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Başlık etiketini oluşturuyoruz
        title_label = tk.Label(title_frame, text="Sanal Kıyafet Deneme Uygulaması", 
                              font=("Arial", 20, "bold"), fg="white", bg="#3498db")
        title_label.pack(pady=10)
        
        # Kontrol çerçevesini oluşturuyoruz
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Butonları yerleştiriyoruz
        self.select_btn = ttk.Button(control_frame, text="Kıyafet Seç", command=self.select_clothing)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.camera_btn = ttk.Button(control_frame, text="Kamerayı Başlat", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        self.test_btn = ttk.Button(control_frame, text="Kıyafeti Test Et", command=self.test_clothing_display)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # Kıyafet dosyasının yolunu gösterecek etiket
        self.path_var = tk.StringVar()
        self.path_var.set("Henüz kıyafet seçilmedi")
        self.path_label = tk.Label(control_frame, textvariable=self.path_var, 
                                  bg="#f0f0f0", fg="#555555")
        self.path_label.pack(side=tk.LEFT, padx=10)
        
        # Durum mesajını gösterecek etiket
        self.status_var = tk.StringVar()
        self.status_var.set("Hazır")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, 
                                    bg="#f0f0f0", fg="#555555")
        self.status_label.pack(pady=5)
        
        # Görüntü önizlemesi için boş bir etiket
        self.preview_label = tk.Label(self.root, bg="black")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Çıkış bilgisi
        info_label = tk.Label(self.root, text="Çıkmak için 'Q' tuşuna basın", 
                             font=("Arial", 10), bg="#f0f0f0", fg="#555555")
        info_label.pack(pady=5)
    
    def select_clothing(self):
        # Kullanıcının bilgisayarından bir kıyafet resmi seçmesini sağlıyoruz
        self.clothing_path = filedialog.askopenfilename(
            initialdir=os.path.expanduser("~/Desktop"),
            title="Kıyafet Görselini Seçin",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*"))
        )
        
        # Seçilen dosya varsa işlemi yap
        if self.clothing_path:
            self.path_var.set(f"Seçilen dosya: {os.path.basename(self.clothing_path)}")
            self.current_clothing = cv2.imread(self.clothing_path, cv2.IMREAD_UNCHANGED)
            
            # Kıyafet dosyası okunamadıysa hata mesajı ver
            if self.current_clothing is None:
                self.status_var.set("HATA: Kıyafet dosyası yüklenemedi!")
            else:
                h, w = self.current_clothing.shape[:2]
                channels = self.current_clothing.shape[2] if len(self.current_clothing.shape) > 2 else 1
                print(f"Yüklenen görüntü: {w}x{h}, {channels} kanal")
                
                # Eğer görselin alfa kanalı yoksa ekle
                if channels == 3:  
                    print("RGB görüntüsü tespit edildi, alpha kanalı ekleniyor...")
                    bgr = self.current_clothing
                    alpha = np.ones((h, w), dtype=bgr.dtype) * 255
                    self.current_clothing = cv2.merge((bgr, alpha[:, :, np.newaxis]))
                
                # Kıyafet başarıyla yüklendi
                self.status_var.set("Kıyafet yüklendi. Kamerayı başlatabilirsiniz.")
    
    def toggle_camera(self):
        # Kamerayı başlat/durdur
        if self.cap is None:
            if self.current_clothing is None:
                self.status_var.set("Lütfen önce bir kıyafet seçin!")
                return
                
            # Kamera açılıyor
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_var.set("HATA: Kamera açılamadı!")
                self.cap = None
                return
                
            # Poz algılama başlatılıyor
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.camera_btn.configure(text="Kamerayı Durdur")
            self.status_var.set("Kamera aktif - Deneme yapılıyor")
            self.process_camera()
        else:
            self.stop_camera()
    
    def stop_camera(self):
        # Kamera durduruluyor
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.pose:
            self.pose.close()
            self.pose = None
        
        # Buton metnini değiştiriyoruz
        self.camera_btn.configure(text="Kamerayı Başlat")
        self.status_var.set("Kamera durduruldu")
        self.preview_label.configure(image=None)
    
    def process_camera(self):
        # Kamera görüntüsünü alıp işliyoruz
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("HATA: Kamera görüntüsü alınamadı!")
            self.stop_camera()
            return
        
        frame = cv2.flip(frame, 1)
        
        # BGR'den RGB'ye dönüştürme
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Poz tespiti işlemi
        results = self.pose.process(rgb_frame)
        
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            # Poz işaretlerini çizmeye başlıyoruz
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
            )
            if self.current_clothing is not None:
                try:
                    # Kıyafeti poz tespitine göre yerleştiriyoruz
                    annotated_frame = self.overlay_clothing(annotated_frame, results.pose_landmarks.landmark)
                except Exception as e:
                    print(f"Kıyafet yerleştirme hatası: {e}")
        
        # İşlenen görüntüyü ekranda gösteriyoruz
        self.display_frame(annotated_frame)
        
        # 'q' tuşuna basıldığında kamerayı durdur
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop_camera()
            return
        
        # Kamera işlemini sürekli yeniliyoruz
        self.root.after(10, self.process_camera)
    
    def overlay_clothing(self, frame, landmarks, clothing_type="shirt"):
        # Kıyafeti vücuda yerleştiriyoruz
        img_h, img_w, _ = frame.shape

        # Omuz ve kalça noktalarını alıyoruz
        l_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * img_w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * img_h))
        r_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img_w),
                      int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img_h))
        
        l_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * img_w),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * img_h))
        r_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * img_w),
                 int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * img_h))
        
        shoulder_width = np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder))
        body_height = np.linalg.norm(((np.array(r_hip) + np.array(l_hip)) / 2) - ((np.array(r_shoulder) + np.array(l_shoulder)) / 2))
        
        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
        
        clothing_copy = self.current_clothing.copy()
        clothing_h, clothing_w = clothing_copy.shape[:2]
        
        # Kıyafet boyutunu ayarlıyoruz
        if clothing_type == "shirt":
            scale_factor = shoulder_width / clothing_w * 1.8
        elif clothing_type == "pants":
            scale_factor = body_height / clothing_h * 2.0  # Pantolon için
        else:
            scale_factor = shoulder_width / clothing_w * 1.8  # Varsayılan
        
        new_width = int(clothing_w * scale_factor)
        new_height = int(clothing_h * scale_factor)
        
        # Kıyafeti yeniden boyutlandırıyoruz
        if new_width > 0 and new_height > 0:
            resized_clothing = cv2.resize(clothing_copy, (new_width, new_height))
            x_offset = shoulder_center[0] - new_width // 2
            y_offset = shoulder_center[1] - int(new_height * 0.2)
            
            if len(resized_clothing.shape) == 2:
                resized_clothing = cv2.cvtColor(resized_clothing, cv2.COLOR_GRAY2BGR)
            
            alpha_channel = None
            if resized_clothing.shape[2] == 4:  # Eğer alpha kanalı varsa
                b, g, r, alpha_channel = cv2.split(resized_clothing)
                resized_clothing_rgb = cv2.merge((b, g, r))  # RGB'ye dönüştür
            else:
                resized_clothing_rgb = resized_clothing

            y1, y2 = max(0, y_offset), min(img_h, y_offset + new_height)
            x1, x2 = max(0, x_offset), min(img_w, x_offset + new_width)
            
            if x1 >= x2 or y1 >= y2:
                return frame
            
            target_region = frame[y1:y2, x1:x2]
            
            crop_y1 = 0 if y_offset >= 0 else -y_offset
            crop_y2 = new_height if y_offset + new_height <= img_h else img_h - y_offset
            crop_x1 = 0 if x_offset >= 0 else -x_offset
            crop_x2 = new_width if x_offset + new_width <= img_w else img_w - x_offset
            
            clothing_part = resized_clothing_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if alpha_channel is not None:
                alpha_part = alpha_channel[crop_y1:crop_y2, crop_x1:crop_x2] / 255.0
                for c in range(3):
                    target_region[:, :, c] = target_region[:, :, c] * (1 - alpha_part) + clothing_part[:, :, c] * alpha_part
                frame[y1:y2, x1:x2] = target_region
            else:
                blended = cv2.addWeighted(target_region, 0.5, clothing_part, 0.7, 0)
                frame[y1:y2, x1:x2] = blended
        
        return frame
    
    def display_frame(self, frame):
        # Görüntüyü RGB'ye dönüştürüp tkinter etiketinde gösteriyoruz
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.png', rgb_frame)[1].tobytes())
        self.current_image = img
        self.preview_label.configure(image=img)
    
    def test_clothing_display(self):
        # Kıyafet testi için yeni bir pencere açıyoruz
        if self.current_clothing is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir kıyafet seçin!")
            return

        test_window = tk.Toplevel(self.root)
        test_window.title("Kıyafet Testi")
        test_window.geometry("400x500")

        channels = self.current_clothing.shape[2] if len(self.current_clothing.shape) > 2 else 1
        info_text = f"Görüntü boyutu: {self.current_clothing.shape[1]}x{self.current_clothing.shape[0]}\n"
        info_text += f"Kanal sayısı: {channels}\n"
        if channels == 4:
            info_text += "Alpha kanalı mevcut (RGBA)"
        else:
            info_text += "Alpha kanalı yok (RGB veya Gri)"

        info_label = tk.Label(test_window, text=info_text)
        info_label.pack(pady=10)

        test_image = self.current_clothing.copy()
        if channels == 4:
            bgr = test_image[:, :, :3]
            alpha = test_image[:, :, 3]

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_img = tk.PhotoImage(data=cv2.imencode('.png', rgb)[1].tobytes())
            alpha_img = tk.PhotoImage(data=cv2.imencode('.png', alpha)[1].tobytes())

            rgb_label = tk.Label(test_window, text="RGB Kanalları:")
            rgb_label.pack()

            rgb_img_label = tk.Label(test_window, image=rgb_img)
            rgb_img_label.image = rgb_img
            rgb_img_label.pack()

            alpha_label = tk.Label(test_window, text="Alpha Kanalı:")
            alpha_label.pack()

            alpha_img_label = tk.Label(test_window, image=alpha_img)
            alpha_img_label.image = alpha_img
            alpha_img_label.pack()
        else:
            if channels == 3:
                rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

            rgb_img = tk.PhotoImage(data=cv2.imencode('.png', rgb)[1].tobytes())
            rgb_img_label = tk.Label(test_window, image=rgb_img)
            rgb_img_label.image = rgb_img
            rgb_img_label.pack()

        if channels == 4:
            background = np.ones((self.current_clothing.shape[0], self.current_clothing.shape[1], 3), dtype=np.uint8) * 200
            alpha = test_image[:, :, 3] / 255.0
            for c in range(3):
                background[:, :, c] = background[:, :, c] * (1 - alpha) + test_image[:, :, c] * alpha

            background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            bg_img = tk.PhotoImage(data=cv2.imencode('.png', background_rgb)[1].tobytes())

            bg_label = tk.Label(test_window, text="Gri arka plan üzerinde:")
            bg_label.pack()

            bg_img_label = tk.Label(test_window, image=bg_img)
            bg_img_label.image = bg_img
            bg_img_label.pack()

    def run(self):
        self.root.mainloop()
        if self.cap is not None:
            self.cap.release()
        if self.pose is not None:
            self.pose.close()

# Uygulama başlatma kodu
if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualTryOn(root)
    app.run()
