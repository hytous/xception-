"""
1.
导入PyTorch库和GUI库
"""
import torch
import torchvision.transforms as transforms
import cv2
import tkinter.font as tkFont
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

# 以下库为加载模型必备库
import numpy as np

from modeling.buildnet import *
from utils.GradCAM import GradCam


"""
2.
加载训练好的模型
使用`torch.load`函数加载它：
"""
model = torch.load(r'D:\S\study\four\graduateDesign\mode\resnet1000\checkpoint.pth.tar')

"""
3.
创建GUI界面
这个示例中，我们使用了`tkinter`、`PIL`和`torchvision`库。
我们首先创建了一个名为`root`的主窗口对象，并定义了一个`image_path`变量来存储所选图像的路径。
"""
root = Tk()
root.title("Fatty Liver Image Classifier")

image_path = ''


def load_image():
    """
    然后我们定义了一个名为`load_image()`的函数，它使用`filedialog`库弹出对话框让用户选择图像文件，
    然后显示所选图像并激活`btn_classify`按钮。
    """
    global image_path
    # 使用filedialog库弹出对话框，让用户选择图像文件
    image_path = filedialog.askopenfilename(initialdir=".", title="Select an Image",
                                            filetypes=(("JPEG files", "*.jpg*"), ("PNG files", "*.png*")))

    # Display selected image
    img = Image.open(image_path)
    # img = img.resize((250, 250))  # 434, 636
    # ImageTk.PhotoImage将图像转换为可以在GUI中显示的图像
    photo = ImageTk.PhotoImage(img)
    # 在GUI上显示选择的图像
    org_image.config(image=photo)
    # 存一下，要不被释放了就不显示了
    org_image.image = photo
    # 激活btn_classify按钮,什么图片都没选择时分类按钮是灰色的，选完图片就可以用了
    btn_classify['state'] = 'normal'  # 工作室状态设为正常，之前是disabled


# gamma变换
def gamma_correction(image, gamma):
    gamma_inv = 1.0 / gamma
    # 生成一个表,表的内容为,像素值对应的gamma变换后的值
    table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)  # 查表并修改图片


def classify_image():
    """
    `classify_image()`函数使用与模型训练时相同的转换操作对所选的图像进行预处理，并使用`torch.no_grad()`
    上下文管理器对模型进行推断并获取预测结果。最后，它更新`label_class`标签以显示所预测的图像类别
    """
    global image_path
    # img = Image.open(image_path)
    img = np.array(Image.open(image_path))  # [480, 640, 3]
    # img = img.resize((3, 434, 636))  # 434, 636
    # 进行gamma变换
    gamma_value = 2
    img = gamma_correction(img, gamma_value)
    # 显示gamma变换后的图像
    img_pil = Image.fromarray(img.astype('uint8'))
    photo = ImageTk.PhotoImage(img_pil)
    enhance_image.config(image=photo)
    enhance_image.image = photo
    # 将图片转为模型需要的格式
    img = np.swapaxes(img, 0, 2)  # 交换第一维和第三维，把色彩通道放到前面
    img = np.swapaxes(img, 1, 2)  # 交换第二维和第三维，把长宽弄正确位置
    # 输入的类型需要为float32
    img = img.astype(np.float32)
    # 转为张量
    img = torch.from_numpy(img)
    # Make predictions on the image
    with torch.no_grad():
        model.eval()
        # 三维转四维，多以维batch_size
        img = img.unsqueeze(0)  # 增一维
        outputs, feature = model(img)
        predicted_class = np.argmax(outputs, axis=1)  # 选出概率最大的作为预测结果
        # predicted_class = torch.argmax(outputs[0])
        predicted_class = int(predicted_class)  # 取出tensor里的数字
        class_names = ['health', 'mild', 'moderate', 'severe']  # 健康 轻度 中度 重度
        predicted_class = class_names[predicted_class]
    # Display predicted class
    label_class.config(text='Predicted class: ' + str(predicted_class))
    gradcam = GradCam(model)
    Gram_img = gradcam.__call__(img)  # 画热力图,结果为tensor
    # 转为pil图片
    Gram_img = Gram_img.squeeze(dim=0)
    img_pil = Image.fromarray((Gram_img.permute(1, 2, 0).numpy()).astype('uint8'))
    photo = ImageTk.PhotoImage(img_pil)
    # 在GUI上显示选择的图像
    Gram_image.config(image=photo)
    # 存一下，要不被释放了就不显示了
    Gram_image.image = photo


def get_model():
    # 定义网络
    my_model = Builder(num_classes=4,  # 4分类
                       backbone='resnet',
                       pretrained=False)
    # 读取训练好的模型
    # checkpoint = torch.load(r'/root/tf-logs/experimentxception_13/checkpoint.pth.tar')
    checkpoint = torch.load(r'D:\S\study\four\graduateDesign\mode\resnet1000\checkpoint.pth.tar')
    # 读取出来的键值最前面多了个module.所以删掉
    checkpoint_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        checkpoint_dict[new_k] = v
    # 载入模型数据
    my_model.load_state_dict(checkpoint_dict)
    return my_model


if __name__ == "__main__":
    """
    在运行GUI应用程序后，点击“Load
    Image”按钮将会弹出一个对话框让用户选择图像文件。选择图像文件后，点击“Classify
    Image”按钮将会对选择的图像进行分类，并在
    `label_class`
    标签上显示结果。
    """
    model = get_model()

    # 创建一个 Frame 控件，用于容纳多个标签控件
    frame = Frame(root)
    frame.pack(side=TOP)
    # 在frame上添加label，label里有图片
    # 原图
    org_image = Label(frame)
    org_image.pack(side=LEFT)
    # 增强后的图像
    enhance_image = Label(frame)
    enhance_image.pack(side=LEFT)
    # 热力图
    Gram_image = Label(frame)
    Gram_image.pack(side=LEFT)

    btn_load = Button(root, text="Load Image", command=load_image)
    btn_load.pack(pady=10)

    btn_classify = Button(root, text="Classify Image", state='disabled', command=classify_image)
    btn_classify.pack(pady=10)

    # 定义字体、大小
    font = tkFont.Font(family="Helvetica", size=20)
    # 在labol中应用该字体
    label_class = Label(root, text='', font=font)
    label_class.pack(pady=10)

    # 循环
    root.mainloop()
