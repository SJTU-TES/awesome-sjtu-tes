import time
import shutil
# import uvicorn
import gradio as gr
from fastapi import FastAPI
from data4co import TSPConcordeSolver, draw_tsp_problem, draw_tsp_solution


TSP_DEFAULT_PATH = "/opt/run/static/tso_default.png"
TSP_PROBLEM_PATH = "/opt/run/media/tsp_problem.png"
TSP_SOLUTION_PATH = "/opt/run/media/tsp_solution.png"


def _handle_tsp_solve(
    file_path: str,
    norm: str,
):
    if file_path is None:
        raise gr.Error("Please upload a '.tsp' file!")
    if norm == '':
        norm = "EUC_2D"
    if norm != "EUC_2D" and norm != "GEO":
        raise gr.Error("Invaild edge_weight_type! Only support 'GEO' and 'EUC_2D'.")
    solver = TSPConcordeSolver(scale=1)
    solver.from_tsp(file_path, norm=norm)
    start_time = time.time()
    solver.solve()
    solved_time = time.time() - start_time
    tours = solver.tours
    points = solver.points
    draw_tsp_problem(
        save_path=TSP_PROBLEM_PATH,
        points=points,
    )
    draw_tsp_solution(
        save_path=TSP_SOLUTION_PATH,
        points=points,
        tours=tours
    )
    message = "Successfully solve the TSP problem, using time ({:.3f}s).".format(solved_time)
    
    return message, TSP_PROBLEM_PATH, TSP_SOLUTION_PATH
    

def handle_tsp_solve(
    file_path: str,
    norm: str,
):
    try:
        message = _handle_tsp_solve(file_path, norm)
    except Exception as e:
        message = str(e)
    return message


def handle_tsp_clear():
    shutil.copy(
        src=TSP_DEFAULT_PATH,
        dst=TSP_PROBLEM_PATH
    )
    shutil.copy(
        src=TSP_DEFAULT_PATH,
        dst=TSP_SOLUTION_PATH
    )
    message = "successfully clear the files!"
    return message, TSP_PROBLEM_PATH, TSP_SOLUTION_PATH


def convert_image_path_to_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


with gr.Blocks() as tsp_page:

    gr.Markdown(
        '''
        This space displays the solution to the TSP problem.

        ## How to use this Space?
        - Upload a '.tsp' file from tsplib .
        - The images of the TSP problem and solution will be shown after you click the solve button.
        - Click the 'clear' button to clear all the files.
        '''
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=7):
            with gr.Row():
                tsp_file = gr.File(
                    file_types=[".tsp"],
                    scale=3
                )
                info = gr.Textbox(
                    value="",
                    label="Log",
                    scale=4,
                )
            norm = gr.Textbox(
                label="Please input the edge_weight_type of the TSP file",
            )
        with gr.Column(scale=4):
            tsp_problem_img = gr.Image(
                value="media/tsp_problem.png", 
                type="filepath", 
                label="TSP Problem", 
            )
        with gr.Column(scale=4):
            tsp_solution_img = gr.Image(
                value="media/tsp_solution.png", 
                type="filepath", 
                label="TSP Solution", 
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
        handle_tsp_solve,
        [tsp_file, norm],
        outputs=[info, tsp_problem_img, tsp_solution_img]
    )
    
    clear_button.click(
        handle_tsp_clear,
        inputs=None,
        outputs=[info, tsp_problem_img, tsp_solution_img]
    )


app = FastAPI()
app = gr.mount_gradio_app(app, tsp_page, path="/")


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860)