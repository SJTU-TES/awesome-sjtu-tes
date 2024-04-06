import time
import shutil
import gradio as gr
from genn_astar import astar


GED_IMG_DEFAULT_PATH = "media/ged_default.png"
GED_SOLUTION_1_PATH = "media/ged_image_1.png"
GED_SOLUTION_2_PATH = "media/ged_image_2.png"
GED_SOLUTION_3_PATH = "media/ged_image_3.png"
GED_SOLUTION_4_PATH = "media/ged_image_4.png"
GED_SOLUTION_5_PATH = "media/ged_image_5.png"
PRETRAINED_PATH = "best_genn_AIDS700nef_gcn_astar.pt"
PRETRAINED_TARGET_PATH = "/home/user/.cache/pygmtools/best_genn_AIDS700nef_gcn_astar.pt"


def _handle_ged_solve(
    gexf_1_path: str,
    gexf_2_path: str
):
    if gexf_1_path is None:
        raise gr.Error("Please upload file completely!")
    if gexf_2_path is None:
        raise gr.Error("Please upload file completely!")
    
    start_time = time.time()
    shutil.move(src=PRETRAINED_PATH, dst=PRETRAINED_TARGET_PATH)
    astar(
        g1_path=gexf_1_path,
        g2_path=gexf_2_path,
        output_path="src",
        filename="ged_image"
    )
    solved_time = time.time() - start_time
    
    message = "Successfully solve the GED problem, using time ({:.3f}s).".format(solved_time)
    
    return message, GED_SOLUTION_1_PATH, GED_SOLUTION_2_PATH, GED_SOLUTION_3_PATH, \
        GED_SOLUTION_4_PATH, GED_SOLUTION_5_PATH
    

def handle_ged_solve(
    gexf_1_path: str,
    gexf_2_path: str
):
    try:
        message = _handle_ged_solve(
            gexf_1_path=gexf_1_path,
            gexf_2_path=gexf_2_path
        )
        return message
    except Exception as e:
        message = str(e)
        return message, GED_SOLUTION_1_PATH, GED_SOLUTION_2_PATH, GED_SOLUTION_3_PATH, \
            GED_SOLUTION_4_PATH, GED_SOLUTION_5_PATH


def handle_ged_clear():
    shutil.copy(
        src=GED_IMG_DEFAULT_PATH,
        dst=GED_SOLUTION_1_PATH
    )
    shutil.copy(
        src=GED_IMG_DEFAULT_PATH,
        dst=GED_SOLUTION_2_PATH
    )
    shutil.copy(
        src=GED_IMG_DEFAULT_PATH,
        dst=GED_SOLUTION_3_PATH
    )
    shutil.copy(
        src=GED_IMG_DEFAULT_PATH,
        dst=GED_SOLUTION_4_PATH
    )
    shutil.copy(
        src=GED_IMG_DEFAULT_PATH,
        dst=GED_SOLUTION_5_PATH
    )

    message = "successfully clear the files!"
    return message, GED_SOLUTION_1_PATH, GED_SOLUTION_2_PATH, GED_SOLUTION_3_PATH, \
            GED_SOLUTION_4_PATH, GED_SOLUTION_5_PATH


def convert_image_path_to_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


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
                ged_img_1 = gr.File(
                    label="Upload .gexf File",
                    file_types=[".gexf"],
                    min_width=40,
                )
                ged_img_2 = gr.File(
                    label="Upload .gexf File",
                    file_types=[".gexf"],
                    min_width=40,
                )
        with gr.Column(scale=2):
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
    with gr.Row(variant="panel"):
        ged_solution_1 = gr.Image(
            value=GED_SOLUTION_1_PATH, 
            type="filepath",
            label="1"
        )
        ged_solution_2 = gr.Image(
            value=GED_SOLUTION_2_PATH, 
            type="filepath", 
            label="2"
        )
        ged_solution_3 = gr.Image(
            value=GED_SOLUTION_3_PATH, 
            type="filepath",
            label="3"
        )
        ged_solution_4 = gr.Image(
            value=GED_SOLUTION_4_PATH, 
            type="filepath", 
            label="4"
        )
        ged_solution_5 = gr.Image(
            value=GED_SOLUTION_5_PATH, 
            type="filepath", 
            label="5"
        )
     
    
    solve_button.click(
        handle_ged_solve,
        [ged_img_1, ged_img_2],
        outputs=[info, ged_solution_1, ged_solution_2,
                 ged_solution_3, ged_solution_4, ged_solution_5]
    )
    
    clear_button.click(
        handle_ged_clear,
        inputs=None,
        outputs=[info, ged_solution_1, ged_solution_2,
                 ged_solution_3, ged_solution_4, ged_solution_5]
    )


if __name__ == "__main__":
    ged_page.launch(debug = True)