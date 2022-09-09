"""Based on google text to speech lib to read text and output sound!

Next step is to read audio file and convert it into text to see it workable or not.

FULL functionality is provided by GOOGLE.
"""
import gtts
from playsound import playsound
import tempfile
import shutil
import os
import speech_recognition as sr
import soundfile
from pydub import AudioSegment


39036060

# tmp_path = tempfile.mkdtemp()
tmp_path = os.path.dirname(os.path.abspath(__file__))


def read_file(file_name):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(cur_path, file_name), 'r', encoding='utf-8') as f:
        text = f.read()
        
    return text


def text_to_speech(file_name, output_speech_file_name='tmp.mp3'):
    text = read_file(file_name)
    
    if "ch" in file_name:
        lang = 'zh-CN'
    else:
        lang = 'en'
    
    tts = gtts.gTTS(text, lang=lang, slow=False)
    
    out_speech_file = os.path.join(tmp_path, output_speech_file_name)
    
    tts.save(out_speech_file)
    
    # convert to real wav file
    
    # sound = AudioSegment.from_mp3(out_speech_file)
    # sound.export(os.path.join(tmp_path, 'tmp.wav'), format="wav")
        
    playsound(out_speech_file)
    
    # convert_audio_file(out_speech_file, tmp_path)

    
def convert_audio_file(file_name, file_path):
    input_file = os.path.join(file_path, file_name)
    output_file = os.path.join(file_path, 'out.wav')
    
    data, samplerate = soundfile.read(input_file)
    soundfile.write(output_file, data, samplerate, subtype='PCM_16')
    print("Convert finished!")

def convert_speech_to_text(speech_file_name, speech_file_path=None):
    recog_obj = sr.Recognizer()
    
    speech_file_path = os.path.join(speech_file_path, speech_file_name)
    
    print(speech_file_path)
    
    with sr.AudioFile(speech_file_path) as source:
        audio_text = recog_obj.listen(source)
        
        try:
            text = recog_obj.recognize_google(audio_text)
            print("Start to convert speech file to text:")
            print(text)
        except:
            print("When try to convert, fail!")
            

if __name__ == "__main__":
    file_name = 'en_text.txt'
    output_speech_file_name = "tmp.wav"
    
    text_to_speech(file_name, output_speech_file_name=output_speech_file_name)
    
    # convert_speech_to_text(output_speech_file_name, speech_file_path=tmp_path)

    # shutil.rmtree(tmp_path)
    print("DONE!")
