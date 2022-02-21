import pandas as pd
import random, re, numpy as np
from collections import namedtuple
#from transformers import AutoTokenizer
import torch
from transformers import BertTokenizer
from collections import defaultdict
import string, math

class Music:
    def __init__(self, config):
        self.is_merge_topic = config.is_merge_topic
        self.is_merge_artist_composer = config.is_merge_artist_composer
        self.playmusic_path = config.playmusic_path # "../input/skillmusic/data/Play Music AI Data.xlsx"
        self.musictopic_path = config.musictopic_path # "../input/skillmusic/data/Music Topic Playlists.xlsx"
        self.by_intents = config.cut_by_intents
        self.data = pd.read_excel(self.playmusic_path, sheet_name="Data Clean 7.3")
        self.train_rate = config.train_rate
        
        self.prepare_template_by_intent()                   
        self.read_all_slot()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
    def read_all_slot(self):
        if not hasattr(self, "synonym_slot_choice"):
            self.synonym_slot_choice = {}
        self.read_synonym()
        self.read_song_artist_slot()
        self.read_topic_slot()
        self.read_sort_by_slot()
        self.read_duration_timepoint_time_slot()
        return
    
    def read_synonym(self):
        synonyms_pd = pd.read_excel(self.playmusic_path, sheet_name="Synonym")
        for column in list(synonyms_pd.columns):
            syn = synonyms_pd[column].replace('', np.nan).dropna().tolist()
            self.synonym_slot_choice[f"<{syn[0]}>"] = syn[1:]
        return
    
    def get_time_slot(self, key):
        templates = self.time_template[key]
        n = len(templates)
        idx = np.random.randint(n)
        template = templates[idx]
        results = []
        
        for w in template.split():
            if w.startswith("@"):
                time = self.random_time(w)
                if np.random.random() > 0.5:
                    time = self.convert_num_to_text(time)
                results.append(str(time))
            else:
                results.append(w)
        string = " ".join(results)
        return string.strip()
    
    def random_time(self, request):
        time = None
        if request == "@hour":
            time = np.random.randint(1, 25)
        elif request == "@min":
            time = np.random.randint(1, 60)
        elif request == "@second":
            time = np.random.randint(1, 60)
        elif request == "@num":
            time = np.random.randint(1, 100)
        else:
            raise ValueError(f"function random_time doesn't support {request}")
        return time

    def convert_num_to_text(self, num):
        if num >= 100 or num < 0:
            raise ValueError(f"function convert_num_to_text doesn't support 0 < number > 100")
        num_dict = {
            0: "không",
            1: "một", 2: "hai",
            3: "ba", 4: "bốn",
            5: "năm", 6: "sáu",
            7: "bảy", 8: "tám",
            9: "chín", 10: "mười"
        }
        
        def get_text(num):
            if num not in num_dict:
                return num_dict[num // 10] + " mươi"
            if num == 0:
                return ""
            return num_dict[num]
        
        converted = None
        if num not in num_dict:
            dozen = (num // 10) * 10
            unit = num % 10
            converted = f"{get_text(dozen)} {get_text(unit)}"
        else:
            converted = get_text(num)
        return converted.strip()

    def prepare_slot_dict(self):
        for template in self.template:
            slots = self.find_slots(template)
            for slot in slots:
                self.update_slot_label_dict(slot)
        return

    def merge_slot_policy(self, slot):
        if self.is_merge_topic == True:     
            if slot.startswith("@topic"):
                slot = "@topic"
        if self.is_merge_artist_composer == True:
            if slot == "@composer":
                slot = "@artist"
        return slot
    
    def update_slot_label_dict(self, word):
        def update(key):
            if key not in self.slot_label_dict:
                self.slot_label_dict[key] = len(self.slot_label_dict)
            #if f"B-{key}" not in self.slot_label_dict:
                #self.slot_label_dict[f"B-{key}"] = len(self.slot_label_dict)
                #self.slot_label_dict[f"I-{key}"] = len(self.slot_label_dict)
            return
            
        if not hasattr(self, "slot_label_dict"):
            self.slot_label_dict = {"else": 0}
        
        if word.startswith("@"):
            word = self.merge_slot_policy(word)
            update(word)
        return
    
    def retrieve_reversed_slot_label_dict(self, val):
        if not hasattr(self, "reversed_slot_label_dict"):
            self.reversed_slot_label_dict = {val:key for key, val in self.slot_label_dict.items()}
        return self.reversed_slot_label_dict[val]
    
    def update_intent_label_dict(self, intent):
        def update(intent):
            if intent not in self.intent_label_dict:
                self.intent_label_dict[intent] = len(self.intent_label_dict)
            return
        
        if not hasattr(self, "intent_label_dict"):
            self.intent_label_dict = {}
            
        update(intent)
        return
    
    def indexed_intent_label(self, intent):
        return self.intent_label_dict[intent]
    
    def item_slot_label(self, key, is_first = True):
        key = self.merge_slot_policy(key)
        if key.startswith("@"):
            return self.slot_label_dict[key]
        """
        if key.startswith("@"):
            if is_first:
                return self.slot_label_dict[f"B-{key}"]
            return self.slot_label_dict[f"I-{key}"]
        """
        return self.slot_label_dict["else"]
    
    def prepare_template_by_intent(self):
        grouped_data = self.data.groupby("Intents")
        self.template_by_intent = {}
        for key, item in grouped_data:
            self.template_by_intent[key] = item["Template "].tolist()
            
        self.intents = self.data["Intents"].tolist()
        self.template = self.data["Template "].tolist()
        return

    def cut_to_small_data(self):
        self.data = self.data_by_intents(self.by_intents)
        self.prepare_template_by_intent()
        #self.to_file_stats(self.template, self.intents, path = "analysis_1/small_")
        return

    def prepare_train_test(self):
        # if ration = 0.3, => train 30%, test 70%
        ratio = self.train_rate
        length_intent = len(self.intents)
        length_template = len(self.template)
        self.train_test_intent = {}
        self.train_test_intent[True] = self.intents[:int(length_intent * ratio)]
        self.train_test_intent[False] = self.intents[int(length_intent * ratio):]

        self.train_test_template = {}
        self.train_test_template[True] = self.template[:int(length_template * ratio)]
        self.train_test_template[False] = self.template[int(length_template * ratio):]

        self.train_test_choices = {True: {}, False: {}}
        for key in self.synonym_slot_choice.keys():
            length = len(self.synonym_slot_choice[key])
            self.train_test_choices[True][key] = self.synonym_slot_choice[key][:int(length * ratio)]
            self.train_test_choices[False][key] = self.synonym_slot_choice[key][int(length * ratio):]
        
        #self.to_file_stats(self.train_test_template[True], self.train_test_intent[True], path = "analysis_1/train_")
        #self.to_file_stats(self.train_test_template[False], self.train_test_intent[False], path = "analysis_1/test_")
        return

    def fill_in_template_simple(self, template, is_train):
        optional_re = r'(\(([^)]+)\))'
        for i in re.findall(optional_re, template):
            w = i[0].replace(" ", "-")
            template = template.replace(i[0], w)

        results = []
        labels = []
        for w in template.split():
            if w.startswith("(") and w.endswith(")"):
                w = w[1:-1]
                choices = w.split("|")
                idx = random.randint(0, len(choices)-1)
                r = self.return_by_key(choices[idx], is_train)
                if r is False:
                    return None,None,None
                if r.strip():
                    results.append(r)
                    for _ in range(len(r.split())):
                        labels.append(choices[idx])
            else:
                r = self.return_by_key(w, is_train)
                if r is False:
                    return None,None,None
                if r.strip():
                    results.append(r)
                    for _ in range(len(r.split())):
                        labels.append(w)
        
        #item_slot_label
        prev = ""
        indexed_labels = []
        for l in labels:
            if l.startswith("@") and prev == l:
                idx = self.item_slot_label(l, False)
            else:
                idx = self.item_slot_label(l, True)
            prev = l
            indexed_labels.append(idx)

        results = " ".join(results)
        results = " ".join([r for r in results.split() if r]) # remove extra space
        assert len(results.split()) == len(labels), print(results, labels)
        return  results.lower(), labels, indexed_labels

    def data_by_intents(self, used_framework):
        data = self.data
        condition = None

        for key in used_framework:
            if condition is None:
                condition = data["Intents"] == key
            else:
                condition = condition | (data["Intents"] == key)

        small_data = data[condition]
        return small_data

    def find_slots(self, template):
        return [w for w in template.split() if w.startswith("@")]

    def find_synonyms(self, template):
        return [w for w in template.split() if w.startswith("<") and w.endswith(">")]

    def find_set_synonyms(self, templates):
        synonyms = set()
        for template in templates:
            synonyms.update(self.find_synonyms(template))
        return synonyms

    def find_set_slots(self, templates):
        slots = set()
        for template in templates:
            slots.update(self.find_slots(template))
        return slots

    def print_stats(self, templates):
        total_templates = len(templates)
        n_sl, n_sy = 0, 0
        max_sl, max_sy = 0, 0
        for template in templates:
            slots = self.find_slots(template)
            synonyms = self.find_synonyms(template)
            n_sl += len(slots)
            n_sy += len(synonyms)
            max_sl = max(max_sl, len(slots))
            max_sy = max(max_sy, len(synonyms))
        print(f"Total Templates: {total_templates} | Avg slots: {n_sl / total_templates} | Avg synonyms: {n_sy / total_templates}\
        | Max slots: {max_sl} | Max synonyms: {max_sy}")
        return

    def count_possible_templates_by_synonyms(self, templates, synonyms):
        cnt = 0
        for template in templates:
            syns = self.find_synonyms(template)
            t = len(syns)
            for syn in syns:
                t = t * len(synonyms[syn])
            cnt += t
        print(f"Total Templates: {len(templates)} | Have variants: {cnt} | Avg: {cnt / len(templates)}")
        return

    def to_file_stats(self, templates, intents, path = ""):
        def write_file(Dict, name):
            with open(name, "w", encoding="utf8") as f:
                f.write(f"Total: {len(Dict)} items \n")
                for key in Dict.keys():
                    f.write(f"{key} - {Dict[key]}\n")
        
        def update_dict(Dict, List):
            for item in List:
                if item not in Dict:
                    Dict[item] = 0
                Dict[item] += 1
            return Dict

        slots = {}
        synonyms = {}
        intents_dict = update_dict(dict(), intents)
        for template in templates:
            slots = update_dict(slots, self.find_slots(template))
            synonyms = update_dict(synonyms, self.find_synonyms(template))

        write_file(slots, path + "slots_stat.txt")
        write_file(synonyms, path + "synonyms_stat.txt")
        write_file(intents_dict, path + "intents_stat.txt")
        return

    def read_topic_slot(self):
        topic_csv = pd.read_excel(self.musictopic_path, sheet_name="Master_Total")
        if self.is_merge_topic == False:
            grouped_data = topic_csv.groupby("\nType")
            topics = defaultdict(lambda : [])
            for key, item in grouped_data:
                for topic in item["Similar name"].tolist():
                    topics[key] += [s.strip() for s in topic.split(",") if s]

            for key in topics.keys():
                topics[key] = list(set(topics[key]))

            # copy from generate-data7.0.py
            self.synonym_slot_choice['@topic-activity'] = topics['activity']
            self.synonym_slot_choice['@topic-activity-mood'] = topics['activity'] + topics['mood']
            self.synonym_slot_choice['@topic-genre-new-mood'] = topics['genre'] + topics['new release'] + topics['mood']
            self.synonym_slot_choice['@topic-season-activity-other'] = topics['season'] + topics['activity'] + topics['other']
            self.synonym_slot_choice['@topic'] = topics['activity'] + topics['mood'] + topics['genre'] + topics['new release'] + topics['season'] + topics['other']
        else:
            topics = topic_csv["Similar name"].tolist()
            topic_slot = []
            for topic in topics:
                topic_slot += [s.strip() for s in topic.split(",") if s]

            self.synonym_slot_choice["@topic"] = list(set(topic_slot))


    def read_song_artist_slot(self):
        path = "../input/skillmusic/data/dump.json"
        song = set()
        artist = set()
        total_skip = 0
        with open(path, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    _, song_title, song_artist = line.replace("\n", "").replace("[", "").replace("]", "").split("\",")
                except:
                    total_skip += 1
                    continue
                if song_title:
                    song.add(song_title.replace("\"", ""))
                if song_artist:
                    artist.add(song_artist.replace("\"", ""))
        self.synonym_slot_choice["@song"] = list(song)
        self.synonym_slot_choice["@artist"] = list(artist)
        
    def read_duration_timepoint_time_slot(self):
        self.time_template = {}
        for key in ["@duration", "@timepoint", "@times"]:
            data = pd.read_excel(self.playmusic_path, sheet_name=key, header=None)
            self.time_template[key] = data[0].tolist()

    def read_sort_by_slot(self):
        data = pd.read_excel(self.playmusic_path, sheet_name="@sort-by")
        columns = list(data.columns)
        a = set()
        for i in [1,5]:
            col = data[columns[i]].replace('', np.nan).dropna().tolist()
            a.update(col)
        self.synonym_slot_choice["@sort_by"] = list(a)

    def choose_index_minimize_repeat(self, key, is_train):
        if not hasattr(self, "choice_tracker"):
            self.choice_tracker = {True:{}, False:{}}
            
        if key not in self.choice_tracker[is_train]:
            self.choice_tracker[is_train][key] = 0
        elif self.choice_tracker[is_train][key] + 1 < len(self.train_test_choices[is_train][key]):
            self.choice_tracker[is_train][key] += 1
        else:
            self.choice_tracker[is_train][key] = 0
        return self.choice_tracker[is_train][key]
    
    def clean_text(self, text):
        template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
        text = template.sub(r'', text)

        table = str.maketrans(dict.fromkeys(string.punctuation)) # remove punctuation
        text = text.translate(table)

        text = re.sub(' +', ' ', text) #Remove Extra Spaces
        text = text.strip()
        return text

    def return_by_key(self, key, is_train):
        key = self.merge_slot_policy(key)
        if key in ["@duration", "@timepoint", "@times"]:
            return self.get_time_slot(key)
        
        if (key.startswith("<") and key.endswith(">")) or (key.startswith("@")):
            if key not in self.train_test_choices[is_train]:
                return False
        else:
            return self.clean_text(key.replace("-", " "))

        choices = self.train_test_choices[is_train][key]
        idx = self.choose_index_minimize_repeat(key, is_train)
        return self.clean_text(choices[idx])

    def generate_train_test_data(self, loop = 10, is_train = True):
        self.train_test_generate = [] # will remove previous
        templates = self.train_test_template[is_train]
        intents = self.train_test_intent[is_train]
        Item = namedtuple("Item", ["text", "bpe", "ids", "slot_label", "pieced_label", "intent", "template"])
        
        def generate_then_append(template, intent, is_train):
            self.update_intent_label_dict(intent)
            text, _, slot_label = self.fill_in_template_simple(template, is_train)
            if text is None:
                return
            try:
                pieced_label, bpe, ids = self.alignment_after_tokenize(text, slot_label)
            except:
                return
            if ids is None:
                return
            indexed_intent = self.indexed_intent_label(intent)
            self.train_test_generate.append(Item(text, bpe, ids, slot_label, pieced_label, indexed_intent, template))
            return
        
        max_num = max([len(item) for key, item in self.template_by_intent.items()])
        for _ in range(loop):
            for intent, templates in self.template_by_intent.items():
                n = len(templates)
                balance_loop = math.ceil(max_num / n)
                for _ in range(balance_loop):
                    for template in templates:
                        generate_then_append(template, intent, is_train)
        return
    
    def check_and_jump(self, tok, arr_text, pieced_label, bpe_flag, i, slot_label):
        if arr_text[i] == tok or tok == "[UNK]":
            pieced_label.append(slot_label[i])
            i += 1
            bpe_flag = False
        elif bpe_flag == True and tok.startswith("##") and tok[2:] in arr_text[i]:
            pieced_label.append(slot_label[i])
        elif arr_text[i].startswith(tok):
            if bpe_flag == False:
                pieced_label.append(slot_label[i])
                bpe_flag = True
            else:
                bpe_flag = False
                return False, bpe_flag, i, slot_label
        else:
            bpe_flag = False
            return False, bpe_flag, i, slot_label
        return True, bpe_flag, i, slot_label
    
    def alignment_after_tokenize(self, text, slot_label):
        bpe = self.tokenizer.tokenize(text)
        i = 0
        pieced_label = [] # because after tokenized, words are extended to word-pieced (dùm = dù ##m)
        bpe_flag = False
        arr_text = text.split()
        
        for tok in bpe:
            r, bpe_flag, i, slot_label = self.check_and_jump(tok, arr_text, pieced_label, bpe_flag, i, slot_label)
            if r == False:
                i += 1
                r, bpe_flag, i, slot_label = self.check_and_jump(tok, arr_text, pieced_label, bpe_flag, i, slot_label)
                if r == False:
                    #print(print(f"Error appear | tok: {tok} | text: {arr_text[i]}"))
                    continue

        #assert len(pieced_label) == len(bpe)
        if len(pieced_label) != len(bpe):
            #print("BPE: ", bpe)
            #print("Text: ", text)
            return None, None, None
        pieced_label = [0] + pieced_label + [0] # add [101] [102] tokens
        indexed_text = self.tokenizer(text, return_tensors='pt')["input_ids"].tolist()[0] # cast 2d array to 1d
        return pieced_label, bpe, indexed_text
    
    def __len__(self):
        assert hasattr(self, "train_test_generate"), print("You should call generate_train_test_data - before call len()")
        return len(self.train_test_generate)
    
    def __getitem__(self, idx):
        assert hasattr(self, "train_test_generate"), print("You should call generate_train_test_data - before call getitem()")
        item = self.train_test_generate[idx]
        return {
            "text": item.text,
            "ids": item.ids,
            "slot_label": item.pieced_label,
            "intent": item.intent,
            "template": item.template
        }
    
    def collate_fn(self, items):
        batch = {
            "text": [],
            "ids": [],
            "mask": [],
            "types": [],            
            "slot_label": [],
            "intent": [],
            "template": []
        }
        max_len = 0
        for item in items:
            n = len(item["slot_label"])
            max_len = max(n, max_len)
            batch["text"].append(item["text"])
            batch["ids"].append(item["ids"])
            batch["mask"].append([1] * n)
            batch["types"].append([0] * n)
            batch["slot_label"].append(item["slot_label"])
            batch["intent"].append(item["intent"])
            batch["template"].append(item["template"])
            
        def padding(data, max_len):
            pad_len = max_len - len(data)
            data = data + [0] * pad_len
            return data
        
        def PaddingData(data, max_len):
            padded_data = [padding(d, max_len) for d in data]
            return padded_data
        
        for key in batch.keys():
            if key not in ["text", "intent", "template"]:
                batch[key] = PaddingData(batch[key], max_len)
            if key not in ["text", "template"]:
                batch[key] = torch.LongTensor(batch[key])
        return batch
    
    def decode_exact_slot(self, ids, slot_label):
        tuple_decoded = namedtuple("Decoded_Slot", ["decoded_ids", "decoded_slot"])
        split_ids = [[]]
        decoded_slot = []
        flag = False
        prev_slot = -1
        for s, i in zip(slot_label, ids):
            if flag == True and s == 0:
                split_ids.append([])
                flag = False
            elif s != 0:
                if flag == False:
                    slot = self.retrieve_reversed_slot_label_dict(s)
                    decoded_slot.append(slot)
                    prev_slot = s
                elif prev_slot != s:
                    prev_slot = s
                    slot = self.retrieve_reversed_slot_label_dict(s)
                    split_ids.append([])
                    decoded_slot.append(slot)
                 
                split_ids[-1].append(i)
                flag = True

        decoded_ids = self.tokenizer.batch_decode(split_ids)
        if not decoded_ids[-1]:
            decoded_ids = decoded_ids[:-1]
        assert len(decoded_ids) == len(decoded_slot)
        return tuple_decoded(decoded_ids, decoded_slot)

    def inference(self, text, model):
        tokenized = self.tokenizer(text)
        max_len = 0
        ids, type_ids, mask = [], [], []
        for i in range(len(text)):
            max_len = max(max_len, len(tokenized["input_ids"][i]))
            ids.append(tokenized["input_ids"][i])
            type_ids.append(tokenized["token_type_ids"][i])
            mask.append(tokenized["attention_mask"][i])

        def padding(data, max_len):
            pad_len = max_len - len(data)
            data = data + [0] * pad_len
            return data

        def PaddingData(data, max_len):
            padded_data = [padding(d, max_len) for d in data]
            return padded_data

        device = model.device
        ids = torch.LongTensor(PaddingData(ids, max_len)).to(device)
        type_ids = torch.LongTensor(PaddingData(type_ids, max_len)).to(device)
        mask = torch.LongTensor(PaddingData(mask, max_len)).to(device)
        batch_size = ids.size(0)
        _, score_intent, score_slot = model(ids, mask, type_ids)
        predict_intent = torch.max(torch.softmax(score_intent, dim=-1),dim=1).indices
        predict_slot = torch.max(torch.softmax(score_slot, dim=-1),dim=-1).indices
        predict_slot = predict_slot.reshape(batch_size,-1)

        for i, [txt, idx, slot_label] in enumerate(zip(text, ids, predict_slot)):
            idx = idx.tolist()
            slot_label = slot_label.tolist()
            print(f"{i+1}. Text: {txt} | {self.decode_exact_slot(idx, slot_label)} | Pred: {slot_label}")
            print("\n")
        return