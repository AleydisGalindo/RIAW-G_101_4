import json
import random


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # statistics table 2
    # fact_algorithms is a dictionary with the selected algorithm counters: key = selected_algorithm | value = algorithm counter
    fact_algorithms = dict([])

    # statistics table 3
    # fact_terms is a dictionary with the query terms counters: key = query term | value = term counter
    fact_terms = dict([])

    # statistics table 4
    # fact_browsers is a dictionary with the browser counters: key = browser | value = browser counter
    fact_browsers = dict([])

    def save_query_terms(self, terms: str) -> int:
        print(self)
        return random.randint(0, 100000)


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return {
            'doc_id': self.doc_id,
            'description': self.description,
            'counter': self.counter
        } # self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
