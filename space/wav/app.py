import whisper
from pydub import AudioSegment
import gradio as gr

def convert_6ch_wav_to_stereo(input_file_path, output_file_path):
    sound = AudioSegment.from_file(input_file_path, format="wav")
    if sound.channels != 6:
        sound.export(output_file_path, format="wav")
        return 
    front_left = sound.split_to_mono()[0]
    front_right = sound.split_to_mono()[1]
    center = sound.split_to_mono()[2]
    back_left = sound.split_to_mono()[4]
    back_right = sound.split_to_mono()[5]
    center = center - 6  
    back_left = back_left - 6  
    back_right = back_right - 6  
    stereo_left = front_left.overlay(center).overlay(back_left)
    stereo_right = front_right.overlay(center).overlay(back_right)
    stereo_sound = AudioSegment.from_mono_audiosegments(stereo_left, stereo_right)
    stereo_sound.export(output_file_path, format="wav")


def judge_command(file_path):
    whisper_model = whisper.load_model("large", device="cpu")
    out_path='./out.wav'
    convert_6ch_wav_to_stereo(file_path,out_path)
    result = whisper_model.transcribe(out_path,language="en")
    text_result = result['text']
    print(text_result)
    return text_result


def handle_audio_transcription(file_path):
    try:
        text_result = judge_command(file_path)
        message = "Transcription successful!"
    except Exception as e:
        message = str(e)
        text_result = ""
    return message, text_result

with gr.Blocks() as audio_transcription_page:

    gr.Markdown(
        '''
        This space transcribes the spoken words from an audio file to text.
        ## How to use this Space?
        - Upload a '.wav' file.
        - The transcription of the audio will be shown after you click the transcribe button.
        '''
    )

    with gr.Row():
        with gr.Column():
            audio_file = gr.File(
                file_types=[".wav"],
                label="Upload a '.wav' file",
            )
            info = gr.Textbox(
                value="",
                label="Log",
                placeholder="Transcription results will appear here...",
            )
        transcribe_button = gr.Button("Transcribe")

    transcribe_button.click(
        handle_audio_transcription,
        [audio_file],
        [info]
    )

if __name__ == "__main__":
    audio_transcription_page.launch(debug=True)
