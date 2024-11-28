from pyinstrument import Profiler
import cProfile
import os


def profile_with_pyinstrument(func, *args, **kwargs):
    profiler = Profiler()
    profiler.start()
    result = func(*args, **kwargs)
    profiler.stop()
    with open("pyinstrument_output.txt", "w") as f:
        f.write(profiler.output_text(unicode=True, color=False))
    with open("pyinstrument_output.html", "w") as f:
        f.write(profiler.output_html())
    return result


def profile_with_cprofile(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    pr.dump_stats("cprofile_output.prof")
    os.system("gprof2dot -f pstats cprofile_output.prof | dot -Tpng -o cprofile_callgraph.png")
    return result
