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
    if video_path is None:
        raise gr.Error("Please upload source video!")
    if img_path is None:
        raise gr.Error("Please upload target image!")
    
    start_time = time.time()
    command = f"python run.py -t {video_path} -s {img_path} -o {ROOP_OUTPUT_VIDEO_PATH}"
    os.system(command)
    solved_time = time.time() - start_time
    message = "Successfully solve the GED problem, using time ({:.3f}s).".format(solved_time)
    
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
    shutil.copy(
        src=ROOP_DEFAULT_PATH,
        dst=ROOP_OUTPUT_VIDEO_PATH
    )

    message = "successfully clear the files!"
    return message, ROOP_OUTPUT_VIDEO_PATH


with gr.Blocks() as ged_page:

    gr.Markdown(
        '''
        This space displays the solution to the Graph Edit Distance problem.
        ## How to use this Space?
        - Upload two '.gexf' files.
        - The images of the GED problem and solution will be shown after you click the solve button.
        - Click the 'clear' button to clear all the files.
        ## Examples
        - You can get the test examples from our [GED Dataset Repo.](https://huggingface.co/datasets/SJTU-TES/Graph-Edit-Distance) 
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