from robot.libraries.BuiltIn import run_keyword_variant
from robot.libraries.BuiltIn import BuiltIn
from robot.api.deco import keyword, library


class KeywordOperation:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    __version__ = '0.1'

    @run_keyword_variant(resolve=2)
    def run_keyword_001(self, name1, name2, *args):
        print(name1)
        ret = BuiltIn.run_keyword(name1, '')
        print(name2)
        if ret:
            BuiltIn.run_keyword(name2, *args)
