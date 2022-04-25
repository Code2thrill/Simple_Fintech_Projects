patterns = {
    'access':['access','lost','account','where'],
    'expiry':['limited','timeframe','expires','expiry','expire','limit','time'],
    'release':['when','release','month','date','day','released'],
    'revisited':['live','revisited','revisit','visit','rewatch'],
    "greeting": ["hi", "hey", "help", "hello"]
}
#关键词的量太少，有时用户会使用关键词的不同时态，或打错字，如何解决？

class IntentClassifier(object):

    def __init__(self, patterns):
        self.patterns = patterns

    def classify(self, text):

        for intent, pattern in self.patterns.items():
# dict.items() returns a list of tuples; in this case (intent, pattern)
            for word in pattern:
#两个for loops太慢了，On^2; 如果patterns多一些运作会很慢,这个时候用Binary search会更快
#还可以count 关键字match的数量，pattens 关键字match text 数量最多的确定 intent
                if word in text.lower():

                    return intent

        return "default"

intent_classifier = IntentClassifier(patterns)

# classification = intent_classifier.classify("How are you?")
# 
# print(classification)

class AnswerGenerator(object):

    def generateAnswer(self, intent):
#这么多if,这要是一个复杂些的bot,该怎么做呢？
        if intent == "greeting":

            return "Hello. Please ask me a question."

        if intent == "access":

            return "We provide our courses online via our training site. Please make an account there to access your courses."

        if intent == "expiry":

            return "Our courses have no expiration date or add-on fees. You will get full lifetime access."

        if intent == "release":

            return "We expect to release this masterclass by May. Our team is dedicated and working hard to release this course as soon as possible."

        if intent == "revisited":

            return "The masterclass can be revisited. It is not live."

        if intent == "default":

            return "I couldn't find an answer to your question. Please email us."

answer_generator = AnswerGenerator()

# answer = answer_generator.generateAnswer("access")
# 
# print(answer)
while True:

    user_input = input('Please enter your question below: \n')

    intent = intent_classifier.classify(user_input)

    answer = answer_generator.generateAnswer(intent)

    print(answer)