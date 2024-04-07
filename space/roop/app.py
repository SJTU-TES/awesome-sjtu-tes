import os
import time
import shutil
import gradio as gr


ROOP_DEFAULT_PATH = "media/roop_default.png"
ROOP_OUTPUT_VIDEO_PATH = "media/roop_output.mp4"


def _handle_roop_solve(
    video_path: str,
    img_path: str
):
    # Check file upload status
    if video_path is None:
        raise gr.Error("Please upload source video!")
    if img_path is None:
        raise gr.Error("Please upload target image!")
    
    # Check if the media folder exists 
    if not os.path.exists("media"):
        os.mkdir("media")
        
    # Begin solve and record the solving time
    start_time = time.time()
    command = f"python run.py -t {video_path} -s {img_path} -o {ROOP_OUTPUT_VIDEO_PATH}"
    os.system(command)
    solved_time = time.time() - start_time
    
    # Message
    message = "Successfully performed face replacement, using time ({:.3f}s).".format(solved_time)
    
    return message, ROOP_OUTPUT_VIDEO_PATH
    

def handle_roop(
    video_path: str,
    img_path: str
):
    try:
        message = _handle_roop_solve(
            video_path=video_path,
            img_path=img_path
        )
        return message
    except Exception as e:
        message = str(e)
        return message, ROOP_OUTPUT_VIDEO_PATH


def handle_roop_clear():
    # Replace the original image with the default image
    shutil.copy(
        src=ROOP_DEFAULT_PATH,
        dst=ROOP_OUTPUT_VIDEO_PATH
    )

    message = "successfully clear the files!"
    return message, ROOP_OUTPUT_VIDEO_PATH


with gr.Blocks() as ged_page:

    gr.Markdown(
        '''
        This space displays how to perform face swapping.
        ## How to use this Space?
        - Upload a video, preferably with a duration of less than 5 seconds.
        - Upload a photo of the person you wish to swap with.
        - You will receive the result of the face swap after 5-10 minutes.
        - Click the 'clear' button to clear all the files.
        ## Examples
        - You can get the test examples from our [Roop Dataset Repo.](https://huggingface.co/datasets/SJTU-TES/Roop) 
        '''
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            with gr.Row():
                upload_video = gr.Video(
                    label="Upload .mp4 Vide0",
                    format="mp4",
                )
                upload_img = gr.Image(
                    label="Upload .png or .jpg File",
                    type="filepath",
                    min_width=40,
                )
            info = gr.Textbox(
                            value="",
                            label="Log",
                            scale=4,
                        )
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    solve_button = gr.Button(
                        value="Solve", 
                        variant="primary", 
                        scale=1
                    )
                with gr.Column(scale=1, min_width=100):
                    clear_button = gr.Button(
                        "Clear", 
                        variant="secondary", 
                        scale=1
                    )
                with gr.Column(scale=8):
                    pass    
        with gr.Column(scale=2):
            output_video = gr.Video(height=405, width=720)     
    
    solve_button.click(
        handle_roop,
        [upload_video, upload_img],
        outputs=[info, output_video]
    )
    
    clear_button.click(
        handle_roop_clear,
        inputs=None,
        outputs=[info, output_video]
    )


if __name__ == "__main__":
    ged_page.launch(debug = True)