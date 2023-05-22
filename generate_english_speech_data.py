import sys
import torch
import soundfile
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
#model = models
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)


text = sys.argv[1]
json_file = sys.argv[2]
error_file = sys.argv[3]


#f_text = open(text, "r")
jsons_sent = open(json_file, "w")
errors_sent = open(error_file, "w")

#lines = f_text.readlines()

i = 19590

for line in open(text, "r"):


    try:
        i += 1
        text = line.split("\t")[0].strip()
        tgt = line.split("\t")[1].strip()
        sample = TTSHubInterface.get_model_input(task, text)
        sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(device)
        sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(device)

        wav, rate = TTSHubInterface.get_prediction(task, models[0].to(device), generator, sample)
        wav = wav.cpu()

        save_dir = "/data/home/yc07490/Russian_GEC/speech/clang8-en/clang8_en_speech/cl8-en-01/cl8_01_en-{}.wav".format(i)
        soundfile.write(save_dir, wav, rate)
        
        text_save_dir = "speech/clang8_en_speech/cl8-en-01/cl8_01_en-{}.wav".format(i)

        jsons = json.dumps({'speech_path': text_save_dir, 'ungrammatical_text': text, 'grammatical_text': tgt})

        jsons_sent.write(jsons + "\n")

    except Exception as ex:

        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print("\n"+message + " :: in 01-files, the Num. {} sentence is wrong.".format(i))
        #print("the Num. {} sentence is wrong".format(i)+ "\t" + line)
        errors_sent.write("The Num. {} sentence is wrong in 01-files: ".format(i) + "\t" + line)
        
    continue
        
