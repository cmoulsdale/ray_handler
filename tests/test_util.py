from ray_handler import handler


def test_subset_dictionary():
    names = ["a", "c"]
    dictionary = dict(a=1, b=2, c=3)
    assert handler.subset_dictionary(names, dictionary) == dict(a=1, c=3)


def test_get_md5():
    """compare to known hash"""

    kwargs = dict(a=1, b=2, c=3)
    md5 = "b6f5884eb9418ccdba005731d88bcb3f"
    assert handler.get_md5(kwargs) == md5
