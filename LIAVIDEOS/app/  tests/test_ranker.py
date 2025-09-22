from ranker import finalize

def test_finalize_non_overlap():
    cands = [
        {"start":0,"end":20,"text":"a","score":0.9},
        {"start":10,"end":30,"text":"b","score":0.8},
        {"start":35,"end":55,"text":"c","score":0.7},
        {"start":60,"end":80,"text":"d","score":0.6},
    ]
    sents = [(0,10,"a1"),(10,20,"a2"),(20,30,"b"),(35,55,"c"),(60,80,"d")]
    out = finalize(cands, sents, max_shorts=2, min_gap=5, max_duration=60)
    assert len(out) == 2
    assert out[0]["start"] >= 0 and out[1]["start"] >= out[0]["end"] + 5
