import re
import copy

class currencyParser:
    """ parse a string of currency into a float

        Args:
            currency_symbol(str): currency_symbol used
    """
    def __init__(self, **kwargs):
        assert("currency_symbol" in kwargs)
        self.currency_symbol = kwargs["currency_symbol"]
    def parse(self, currency):
        """ parse currency

        Args:
            currency(string): string of currency value. For example:


        Returns: float value of the currency.

        Example:
            "$5,000.01" -> 5000.01
        """
        regex = "[{},]".format(self.currency_symbol)
        return float(re.sub(regex, '', currency))

class stringParser:
    def __init__(self, **kwargs):
        pass
    def parse(self, string):
        return string

class integerParser:
    """ parse a string of integer into an int

        - remove thousand seperator
    """
    def __init__(self, **kwargs):
        pass
    def parse(self, integer):
        """

        Args:
            integer(str): string of the integer

        Returns:
            int:

        Example:
            "2,000" -> 2000
        """
        return int(integer.replace(",", ""))

TYPE_TO_PARSER = {"string": stringParser, "integer": integerParser}

def parse_data(data, schema):
    """ parse the data based on schema

    Args:
        data(pandas.DataFrame): a pandas.DataFrame where dtype are all strings
        schema(dict): dictionary of fields with types and other information

    Returns:
        pandas.DataFrame: pandas.DataFrame with desired parsing logic according to schema

    Example:
        schema = {"col1": {"type": "string"}, "col2": {"type":"float", "digits": 2}}
    """
    parsers = {}
    schema = copy.deepcopy(schema)
    for col in schema:
        type = schema[col].pop("type")
        arguments = schema[col]
        parsers[col] = TYPE_TO_PARSER[type](**arguments)
        data[col] = data[col].apply(parsers[col].parse)

    return data


