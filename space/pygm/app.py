import time
import shutil
import gradio as gr
from pygm_rrwm import pygm_rrwm


PYGM_IMG_DEFAULT_PATH = "src/pygm_default.png"
PYGM_SOLUTION_1_PATH = "src/pygm_image_1.png"
PYGM_SOLUTION_2_PATH = "src/pygm_image_2.png"


def _handle_pygm_solve(
    img_1_path: str,
    img_2_path: str,
    kpts1_path: str,
    kpts2_path: str,
):
    if img_1_path is None:
        raise gr.Error("Please upload file completely!")
    if img_2_path is None:
        raise gr.Error("Please upload file completely!")
    if kpts1_path is None:
        raise gr.Error("Please upload file completely!")
    if kpts1_path is None:
        raise gr.Error("Please upload file completely!")
    
    start_time = time.time()
    pygm_rrwm(
        img1_path=img_1_path,
        img2_path=img_2_path,
        kpts1_path=kpts1_path,
        kpts2_path=kpts2_path,
        output_path="src",
        filename="pygm_image"
    )
    solved_time = time.time() - start_time
    
    message = "Successfully solve the TSP problem, using time ({:.3f}s).".format(solved_time)
    
    return message, PYGM_SOLUTION_1_PATH, PYGM_SOLUTION_2_PATH
    

def handle_pygm_solve(
    img_1_path: str,
    img_2_path: str,
    kpts1_path: str,
    kpts2_path: str,
):
    try:
        message = _handle_pygm_solve(
            img_1_path=img_1_path,
            img_2_path=img_2_path,
            kpts1_path=kpts1_path,
            kpts2_path=kpts2_path,
        )
        return message
    except Exception as e:
        message = str(e)
        return message, PYGM_SOLUTION_1_PATH, PYGM_SOLUTION_2_PATH


def handle_pygm_clear():
    shutil.copy(
        src=PYGM_IMG_DEFAULT_PATH,
        dst=PYGM_SOLUTION_1_PATH
    )
    shutil.copy(
        src=PYGM_IMG_DEFAULT_PATH,
        dst=PYGM_SOLUTION_2_PATH
    )

    message = "successfully clear the files!"
    return message, PYGM_SOLUTION_1_PATH, PYGM_SOLUTION_2_PATH


def convert_image_path_to_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


with gr.Blocks() as pygm_page:

    gr.Markdown(
        '''
        This space displays the solution to the Graph Matching problem.
        ## How to use this Space?
        - Upload a '.pygm' file from pygmlib .
        - The images of the TSP problem and solution will be shown after you click the solve button.
        - Click the 'clear' button to clear all the files.
        '''
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            with gr.Row():
                pygm_img_1 = gr.File(
                    label="Upload .png File",
                    file_types=[".png"],
                    min_width=40,
                )
                pygm_img_2 = gr.File(
                    label="Upload .png File",
                    file_types=[".png"],
                    min_width=40,
                )
            with gr.Row():
                pygm_kpts_1 = gr.File(
                    label="Upload .mat File",
                    file_types=[".mat"],
                    min_width=40,
                )
                pygm_kpts_2 = gr.File(
                    label="Upload .mat File",
                    file_types=[".mat"],
                    min_width=40,
                )
            info = gr.Textbox(
                value="",
                label="Log",
                scale=4,
            )
        with gr.Column(scale=2):
            pygm_solution_1 = gr.Image(
                value=PYGM_SOLUTION_1_PATH, 
                type="filepath",
                label="Original Images"
            )
            pygm_solution_2 = gr.Image(
                value=PYGM_SOLUTION_2_PATH, 
                type="filepath", 
                label="Graph Matching Results"
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
    
    solve_button.click(
        handle_pygm_solve,
        [pygm_img_1, pygm_img_2, pygm_kpts_1, pygm_kpts_2],
        outputs=[info, pygm_solution_1, pygm_solution_2]
    )
    
    clear_button.click(
        handle_pygm_clear,
        inputs=None,
        outputs=[info, pygm_solution_1, pygm_solution_2]
    )


if __name__ == "__main__":
    pygm_page.launch(debug = True)