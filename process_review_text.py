import wordninja

class process_text():
    def __init__():

        self.contraction_mapping = {
        "ain't": "is not",               "aren't": "are not",               "can't": "cannot",               "'cause": "because",
        "could've": "could have",        "couldn't": "could not",           "didn't": "did not",             "doesn't": "does not",
        "don't": "do not",               "hadn't": "had not",               "hasn't": "has not",             "haven't": "have not",
        "he'd": "he would",              "he'll": "he will",                "he's": "he is",                 "how'd": "how did",
        "how'd'y": "how do you",         "how'll": "how will",              "how's": "how is",               "i'd": "i would",
        "i'd've": "i would have",        "i'll": "i will",                  "i'll've": "i will have",        "i'm": "i am",
        "i've": "i have",                "i'd": "i would",                  "i'd've": "i would have",        "i'll": "i will",
        "i'll've": "i will have",        "i'm": "i am",                     "i've": "i have",                "isn't": "is not",
        "it'd": "it would",              "it'd've": "it would have",        "it'll": "it will",              "it'll've": "it will have",
        "it's": "it is",                 "let's": "let us",                 "ma'am": "madam",                "mayn't": "may not",
        "might've": "might have",        "mightn't": "might not",           "mightn't've": "might not have", "must've": "must have",
        "mustn't": "must not",           "mustn't've": "must not have",     "needn't": "need not",           "needn't've": "need not have",
        "o'clock": "of the clock",       "oughtn't": "ought not",           "oughtn't've": "ought not have", "shan't": "shall not",
        "sha'n't": "shall not",          "shan't've": "shall not have",     "she'd": "she would",            "she'd've": "she would have",
        "she'll": "she will",            "she'll've": "she will have",      "she's": "she is",               "should've": "should have",
        "shouldn't": "should not",       "shouldn't've": "should not have", "so've": "so have",              "so's": "so as",
        "this's": "this is",             "that'd": "that would",            "that'd've": "that would have",  "that's": "that is",
        "there'd": "there would",        "there'd've": "there would have",  "there's": "there is",           "here's": "here is",
        "they'd": "they would",          "they'd've": "they would have",    "they'll": "they will",          "they'll've": "they will have",
        "they're": "they are",           "they've": "they have",            "to've": "to have",              "wasn't": "was not",
        "we'd": "we would",              "we'd've": "we would have",        "we'll": "we will",              "we'll've": "we will have",
        "we're": "we are",               "we've": "we have",                "weren't": "were not",           "what'll": "what will",
        "what'll've": "what will have",  "what're": "what are",             "what's": "what is",             "what've": "what have",
        "when's": "when is",             "when've": "when have",            "where'd": "where did",          "where's": "where is",
        "where've": "where have",        "who'll": "who will",              "who'll've": "who will have",    "who's": "who is",
        "who've": "who have",            "why's": "why is",                 "why've": "why have",            "will've": "will have",
        "won't": "will not",             "won't've": "will not have",       "would've": "would have",        "wouldn't": "would not",
        "wouldn't've": "would not have", "y'all": "you all",                "y'all'd": "you all would",      "y'all'd've": "you all would have",
        "y'all're": "you all are",       "y'all've": "you all have",        "you'd": "you would",            "you'd've": "you would have",
        "you'll": "you will",            "you'll've": "you will have",      "you're": "you are",             "you've": "you have" 
        }


        self.stopwords = set([
            "i", "me", "my", "myself", "we", "our"
            , "ours", "ourselves", "you", "your", "yours", "yourself"
            , "yourselves", "he", "him", "his", "himself", "she"
            , "her", "hers", "herself", "it", "its", "itself"
            , "they", "them", "their", "theirs", "themselves", "what"
            , "which", "who", "whom", "this", "that", "these"
            , "those", "am", "is", "are", "was", "were"
            , "be", "been", "being", "have", "has", "had"
            , "having", "do", "does", "did", "doing", "a"
            , "an", "the", "and", "but", "if", "or"
            , "because", "as", "until", "while", "of", "at"
            , "by", "for", "with", "about", "against", "between"
            , "into", "through", "during", "before", "after", "above"
            , "below", "to", "from", "up", "down", "in"
            , "out", "on", "off", "over", "under", "again"
            , "further", "then", "once", "here", "there", "when"
            , "where", "why", "how", "all", "any", "both"
            , "each", "few", "more", "most", "other", "some"
            , "such", "no", "nor", "not", "only", "own"
            , "same", "so", "than", "too", "very", "s"
            , "t", "can", "will", "just", "don", "should", "now"
            ])

        self.punct_mapping = {
                              "‘": "'"        , "₹": "rupee" , "´": "'", 
                              "°": ""         , "€": "e"     , "™": "tm", 
                              "√": " sqrt "   , "×": "x"     , "²": "2", 
                              "—": "-"        , "–": "-"     , "’": "'", 
                              "_": "-"        , "`": "'"     , '“': '"', 
                              '”': '"'        , '“': '"'     , "£": "e", 
                              '∞': 'infinity' , 'θ': 'theta' , '÷': '/', 
                              'α': 'alpha'    , '•': '.'     , 'à': 'a', 
                              '−': '-'        , 'β': 'beta'  , '∅': '', 
                              '³': '3'        , 'π': 'pi' 
                              }

        self.punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        self.mispell_dict = {'colour'        : 'color'            ,   'centre'        : 'center'           ,
                             'favourite'     : 'favorite'         ,   'travelling'    : 'traveling'        ,
                             'counselling'   : 'counseling'       ,   'theatre'       : 'theater'          ,
                             'cancelled'     : 'canceled'         ,   'labour'        : 'labor'            ,
                             'organisation'  : 'organization'     ,   'citicise'      : 'criticize'        ,
                             'youtu'         : 'youtube '         ,   'qoura'         : 'quora'            ,
                             'sallary'       : 'salary'           ,   'whta'          : 'what'             ,
                             'narcisist'     : 'narcissist'       ,   'howdo'         : 'how do'           ,
                             'whatare'       : 'what are'         ,   'howcan'        : 'how can'          ,
                             'howmuch'       : 'how much'         ,   'howmany'       : 'how many'         ,
                             'whydo'         : 'why do'           ,   'demonetisation': 'demonetization'   ,
                             'thebest'       : 'the best'         ,   'howdoes'       : 'how does'         ,
                             'etherium'      : 'ethereum'         ,   'narcissit'     : 'narcissist'       ,
                             'bigdata'       : 'big data'         ,   '2k17'          : '2017'             ,
                             '2k18'          : '2018'             ,   'qouta'         : 'quota'            ,
                             'exboyfriend'   : 'ex boyfriend'     ,   'airhostess'    : 'air hostess'      ,
                             'whst'          : 'what'             ,   'watsapp'       : 'whatsapp'         ,
                             'demonitisation': 'demonetization'   ,   'demonitization': 'demonetization'   
                             }

    def clean_special_chars(self, text):
        for p in self.punct_mapping:
            text = text.replace(p, mapping[p])
        for p in self.punct:
            text = text.replace(p, f' {p} ')
        return text

    def clean_contractions(self, text):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([self.contraction_mapping.get(t,t) for t in text.split()])
        return text

    def correct_spelling(self, text):
        for word in self.mispell_dict.keys():
            text = text.replace(word, dic[word])
        return text

    def remove_stopwords(self, text):
        text = [i for i in text.split() if i not in self.stopwords]
        text = [wordninja.split(word) for word in text]
        return " ".join(text)

    def clean_text(self, text):
        text = " "+text.lower()+" "
        text = self.clean_contractions(text)
        text = self.clean_special_chars(text)
        text = self.correct_spelling(text)
        text = self.remove_stopwords(text)
        return text.strip()