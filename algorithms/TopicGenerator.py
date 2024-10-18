import logging
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired


class TopicGenerator:
    logger = logging.getLogger('TopicGenerator')

    def __init__(self, docs):
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.docs = docs

    def topic_modeling(self):
        mmr = MaximalMarginalRelevance(diversity=0.3)
        self.topic_model = BERTopic(language="multilingual", representation_model=mmr)
        self.topics, self.probs = self.topic_model.fit_transform(self.docs)
        return self.topic_model, self.topics, self.probs

