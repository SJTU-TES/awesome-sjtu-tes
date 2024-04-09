import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import gradio as gr
from src import model
from src import util
from src.body import Body
from src.hand import Hand


def pose_estimation(test_image):
    bgr_image_path = './test.png'
    with open(bgr_image_path, 'wb') as bgr_file:
        bgr_file.write(test_image)
    # 加载估计模型
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')

    test_image = bgr_image_path
    oriImg = cv2.imread(test_image)  # B,G,R order

    # oriImg = test_image

    # 姿态估计
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    # 绘制身体姿态
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # print(candidate)
    # print(subset)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.savefig('./out.jpg')
    # plt.show()
    return './out.jpg'

# Convert the image path to bytes for Gradio to display
def convert_image_to_bytes(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Pose Estimation")
    with gr.Row():
        image = gr.File(label="Upload Image", type="binary")
        output_image = gr.Image(label="Estimation Result")
    submit_button = gr.Button("Start Estimation")
    
    # Run pose estimation and display results when the button is clicked
    submit_button.click(
        pose_estimation,
        inputs=[image],
        outputs=[output_image]
    )
    
    # Clear the results
    clear_button = gr.Button("Clear")
    def clear_outputs():
        output_image.clear()
    clear_button.click(
        clear_outputs,
        inputs=[],
        outputs=[output_image]
    )

if __name__ == "__main__":
    demo.launch(debug=True)