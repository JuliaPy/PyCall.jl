# GUI event loops and toolkit integration for Python, most importantly
# to support plotting in a window without blocking.  Currently, we
# support wxWidgets, Qt4, Qt5, Qt6, and GTK+.

############################################################################

# global variable to specify default GUI toolkit to use
gui = :default # one of :default, :wx, :qt, :tk, or :gtk

pyexists(mod) = try
    pyimport(mod)
    true
  catch
    false
  end

# Tkinter was renamed to tkinter in Python 3
function tkinter_name()
    return pyversion.major < 3 ? "Tkinter" : "tkinter"
end

pygui_works(gui::Symbol) = gui == :default ||
    ((gui == :wx && pyexists("wx")) ||
     (gui == :gtk && pyexists("gtk")) ||
     (gui == :gtk3 && pyexists("gi")) ||
     (gui == :tk && pyexists(tkinter_name())) ||
     (gui == :qt_pyqt4 && pyexists("PyQt4")) ||
     (gui == :qt_pyqt5 && pyexists("PyQt5")) ||
     (gui == :qt_pyqt6 && pyexists("PyQt6")) ||
     (gui == :qt_pyside && pyexists("PySide")) ||
     (gui == :qt_pyside2 && pyexists("PySide2")) ||
     (gui == :qt_pyside6 && pyexists("PySide6")) ||
     (gui == :qt4 && (pyexists("PyQt4") || pyexists("PySide"))) ||
     (gui == :qt5 && (pyexists("PyQt5") || pyexists("PySide2"))) ||
     (gui == :qt6 && (pyexists("PyQt6") || pyexists("PySide6"))) ||
     (gui == :qt && (pyexists("PyQt6") || pyexists("PyQt5") || pyexists("PyQt4") || pyexists("PySide") || pyexists("PySide2") || pyexists("PySide6"))))

# get or set the default GUI; doesn't affect running GUI
"""
    pygui()

Return the current GUI toolkit as a symbol.
"""
function pygui()
    global gui
    if gui::Symbol != :default && pygui_works(gui::Symbol)
        return gui::Symbol
    else
        for g in (:tk, :qt, :wx, :gtk, :gtk3)
            if pygui_works(g)
                gui = g
                return gui::Symbol
            end
        end
        error("No supported Python GUI toolkit is installed.")
    end
end
function pygui(g::Symbol)
    global gui
    if g != gui::Symbol
        if !(g in (:wx,:gtk,:gtk3,:tk,:qt,:qt4,:qt5,:qt6,:qt_pyqt4,:qt_pyqt5,:qt_pyqt6,:qt_pyside,:qt_pyside2,:qt_pyside6,:default))
            throw(ArgumentError("invalid gui $g"))
        elseif !pygui_works(g)
            error("Python GUI toolkit for $g is not installed.")
        end
        gui = g
    end
    return g::Symbol
end

############################################################################
# Event loops for various toolkits.

# call doevent(status) every sec seconds
function install_doevent(doevent, sec::Real)
    return Base.Timer(doevent, sec, interval=sec)
end

# For PyPlot issue #181: recent pygobject releases emit a warning
# if we don't specify which version we want:
function gtk_requireversion(gtkmodule::AbstractString, vers::VersionNumber=v"3.0")
    if startswith(gtkmodule, "gi.")
        gi = pyimport("gi")
        if gi.get_required_version("Gtk") === nothing
            gi.require_version("Gtk", string(vers))
        end
    end
end

# GTK (either GTK2 or GTK3, depending on gtkmodule):
function gtk_eventloop(gtkmodule::AbstractString, sec::Real=50e-3)
    gtk_requireversion(gtkmodule)
    gtk = pyimport(gtkmodule)
    events_pending = gtk."events_pending"
    main_iteration = gtk."main_iteration"
    install_doevent(sec) do async
        # handle all pending
        while pycall(events_pending, Bool)
            pycall(main_iteration, PyObject)
        end
    end
end

# As discussed in PyPlot.jl#278, Qt looks for a file qt.conf in
# the same path as the running executable, which tells it where
# to find plugins etcetera.  For Python's Qt, however, this will
# be in the path of the python executable, not julia.  Furthermore,
# we can't copy it to the location of julia (even if that location
# is writable) because that would assume that all julia programs
# use the same version of Qt (e.g. via the same Python), which
# is not necessarily the case, and even then it wouldn't work
# for other programs linking libjulia.  Unfortunately, there
# seems to be no way to change this.  However, we can at least
# use set QT_PLUGIN_PATH by parsing qt.conf ourselves, and
# this seems to fix some of the path-related problems on Windows.
# ... unfortunately, it seems fixqtpath has to be called before
# the Qt library is loaded.
function fixqtpath(qtconf=joinpath(dirname(pyprogramname),"qt.conf"))
    haskey(ENV, "QT_PLUGIN_PATH") && return false
    if isfile(qtconf)
        for line in eachline(qtconf)
            m = match(r"^\s*prefix\s*=(.*)$"i, line)
            if m !== nothing
                dir = strip(m.captures[1])
                if startswith(dir, '"') && endswith(dir, '"')
                    dir = dir[2:end-1]
                end
                plugin_path = joinpath(dir, "plugins")
                if isdir(plugin_path)
                    # for some reason I don't understand,
                    # if libpython has already been loaded, then
                    # we need to use Python's setenv rather than Julia's
                    PyDict(pyimport("os")."environ")["QT_PLUGIN_PATH"] = realpath(plugin_path)
                    return true
                end
            end
        end
    end
    return false
end

# Qt: (PyQt6, PyQt5, PyQt4, or PySide module)
function qt_eventloop(QtCore::PyObject, sec::Real=50e-3)
    fixqtpath()
    instance = QtCore."QCoreApplication"."instance"
    AllEvents = QtCore."QEventLoop"."AllEvents"
    processEvents = QtCore."QCoreApplication"."processEvents"
    pop!(ENV, "QT_PLUGIN_PATH", "") # clean up environment
    maxtime = PyObject(50)
    install_doevent(sec) do async
        app = pycall(instance, PyObject)
        if !(app ≛ pynothing[])
            app."_in_event_loop" = true
            pycall(processEvents, PyObject, AllEvents, maxtime)
        end
    end
end

qt_eventloop(QtModule::AbstractString, sec::Real=50e-3) =
    qt_eventloop(pyimport("$QtModule.QtCore"), sec)

function qt_eventloop(sec::Real=50e-3)
    for QtModule in ("PyQt6", "PyQt5", "PyQt4", "PySide", "PySide2", "PySide6")
        try
            return qt_eventloop(QtModule, sec)
        catch
        end
    end
    error("no Qt module found")
end

# wx:  (based on IPython/lib/inputhookwx.py, which is 3-clause BSD-licensed)
function wx_eventloop(sec::Real=50e-3)
    wx = pyimport("wx")
    GetApp = wx."GetApp"
    EventLoop = wx."EventLoop"
    EventLoopActivator = wx."EventLoopActivator"
    install_doevent(sec) do async
        app = pycall(GetApp, PyObject)
        if !(app ≛ pynothing[])
            app."_in_event_loop" = true
            evtloop = pycall(EventLoop, PyObject)
            ea = pycall(EventLoopActivator, PyObject, evtloop)
            Pending = evtloop."Pending"
            Dispatch = evtloop."Dispatch"
            while pycall(Pending, Bool)
                pycall(Dispatch, PyObject)
            end
            pydecref(ea) # deactivate event loop
            pycall(app."ProcessIdle", PyObject)
        end
    end
end

# Tk: (Tkinter/tkinter module)
# based on https://github.com/ipython/ipython/blob/7.0.1/IPython/terminal/pt_inputhooks/tk.py
function Tk_eventloop(sec::Real=50e-3)
    Tk = pyimport(tkinter_name())
    _tkinter = pyimport("_tkinter")
    flag = _tkinter.ALL_EVENTS | _tkinter.DONT_WAIT
    root = PyObject(nothing)
    install_doevent(sec) do async
        new_root = Tk."_default_root"
        if !(new_root ≛ pynothing[])
            root = new_root
        end
        if !(root ≛ pynothing[])
            while pycall(root."dooneevent", Bool, flag)
            end
        end
    end
end
# cache running event loops (so that we don't start any more than once)
const eventloops = Dict{Symbol,Timer}()

"""
    pygui_start(gui::Symbol = pygui())

Start the event loop of a certain toolkit.

The argument `gui` defaults to the current default GUI, but it could be `:wx`, `:gtk`, `:gtk3`, `:tk`, or `:qt`.

"""
function pygui_start(gui::Symbol=pygui(), sec::Real=50e-3)
    pygui(gui)
    if !haskey(eventloops, gui)
        if gui == :wx
            eventloops[gui] = wx_eventloop(sec)
        elseif gui == :gtk
            eventloops[gui] = gtk_eventloop("gtk", sec)
        elseif gui == :gtk3
            eventloops[gui] = gtk_eventloop("gi.repository.Gtk", sec)
        elseif gui == :tk
            eventloops[gui] = Tk_eventloop(sec)
        elseif gui == :qt_pyqt4
            eventloops[gui] = qt_eventloop("PyQt4", sec)
        elseif gui == :qt_pyqt5
            eventloops[gui] = qt_eventloop("PyQt5", sec)
        elseif gui == :qt_pyqt6
            eventloops[gui] = qt_eventloop("PyQt6", sec)
        elseif gui == :qt_pyside
            eventloops[gui] = qt_eventloop("PySide", sec)
        elseif gui == :qt_pyside2
            eventloops[gui] = qt_eventloop("PySide2", sec)
        elseif gui == :qt_pyside6
            eventloops[gui] = qt_eventloop("PySide6", sec)
        elseif gui == :qt4
            try
                eventloops[gui] = qt_eventloop("PyQt4", sec)
            catch
                eventloops[gui] = qt_eventloop("PySide", sec)
            end
        elseif gui == :qt5
            try
                eventloops[gui] = qt_eventloop("PyQt5", sec)
            catch
                eventloops[gui] = qt_eventloop("PySide2", sec)
            end
        elseif gui == :qt6
            try
                eventloops[gui] = qt_eventloop("PyQt6", sec)
            catch
                eventloops[gui] = qt_eventloop("PySide6", sec)
            end
        elseif gui == :qt
            eventloops[gui] = qt_eventloop(sec)
        else
            throw(ArgumentError("unsupported GUI type $gui"))
        end
    end
    gui
end

"""
    pygui_stop(gui::Symbol = pygui())

Stop any running event loop for gui. The `gui` argument defaults to current default GUI.

"""
function pygui_stop(gui::Symbol=pygui())
    if haskey(eventloops, gui)
        Base.close(pop!(eventloops, gui))
        true
    else
        false
    end
end

pygui_stop_all() = for gui in keys(eventloops); pygui_stop(gui); end

############################################################################
