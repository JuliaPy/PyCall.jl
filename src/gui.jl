# GUI event loops and toolkit integration for Python, most importantly
# to support plotting in a window without blocking.  Currently, we
# support wxWidgets, Qt4, and GTK+.

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
    return  pyversion < v"3" ? "Tkinter" : "tkinter"
end

pygui_works(gui::Symbol) = gui == :default ||
    ((gui == :wx && pyexists("wx")) ||
     (gui == :gtk && pyexists("gtk")) ||
     (gui == :gtk3 && pyexists("gi.repository.Gtk")) ||
     (gui == :tk && pyexists(tkinter_name())) ||
     (gui == :qt_pyqt4 && pyexists("PyQt4")) ||
     (gui == :qt_pyside && pyexists("PySide")) ||
     (gui == :qt && (pyexists("PyQt4") || pyexists("PySide"))))
     
# get or set the default GUI; doesn't affect running GUI
function pygui()
    global gui
    if gui::Symbol != :default && pygui_works(gui::Symbol)
        return gui::Symbol
    else
        for g in (:tk, :qt, :wx, :gtk, :gtk3)
            if pygui_works(g)
                gui::Symbol = g
                return gui::Symbol
            end
        end
        error("No supported Python GUI toolkit is installed.")
    end
end
function pygui(g::Symbol)
    global gui
    if g != gui::Symbol
        if !(g in (:wx,:gtk,:gtk3,:tk,:qt,:qt_pyqt4,:qt_pyside,:default))
            throw(ArgumentError("invalid gui $g"))
        elseif !pygui_works(g)
            error("Python GUI toolkit for $g is not installed.")
        end
        gui::Symbol = g
    end
    return g
end

############################################################################
# Event loops for various toolkits.

# call doevent(status) every sec seconds
function install_doevent(doevent::Function, sec::Real)
    timeout = Base.Timer(doevent,sec,sec)
    return timeout
end

# work around API change in Julia 0.3 (issue #76)
if VERSION >= v"0.3-"
    macro doevent(body)
        Expr(:function, :($(esc(:doevent))(async)), body)
    end
else
    macro doevent(body)
        Expr(:function, :($(esc(:doevent))(async, status::Int32)), body)
    end
end

# GTK (either GTK2 or GTK3, depending on gtkmodule):
function gtk_eventloop(gtkmodule::AbstractString, sec::Real=50e-3)
    gtk = pyimport(gtkmodule)
    events_pending = gtk["events_pending"]
    main_iteration = gtk["main_iteration"]
    @doevent begin
        # handle all pending
        while pycall(events_pending, Bool)
            pycall(main_iteration, PyObject)
        end
    end
    install_doevent(doevent, sec)
end

# Qt4: (PyQt4 or PySide module)
function qt_eventloop(QtModule="PyQt4", sec::Real=50e-3)
    QtCore = pyimport("$QtModule.QtCore")
    instance = QtCore["QCoreApplication"]["instance"]
    AllEvents = QtCore["QEventLoop"]["AllEvents"]
    processEvents = QtCore["QCoreApplication"]["processEvents"]
    maxtime = PyObject(50)
    @doevent begin
        app = pycall(instance, PyObject)
        if app.o != pynothing
            app["_in_event_loop"] = true
            pycall(processEvents, PyObject, AllEvents, maxtime)
        end
    end
    install_doevent(doevent, sec)
end

# wx:  (based on IPython/lib/inputhookwx.py, which is 3-clause BSD-licensed) 
function wx_eventloop(sec::Real=50e-3)
    wx = pyimport("wx")
    GetApp = wx["GetApp"]
    EventLoop = wx["EventLoop"]
    EventLoopActivator = wx["EventLoopActivator"]
    @doevent begin
        app = pycall(GetApp, PyObject)
        if app.o != pynothing
            app["_in_event_loop"] = true
            evtloop = pycall(EventLoop, PyObject)
            ea = pycall(EventLoopActivator, PyObject, evtloop)
            Pending = evtloop["Pending"]
            Dispatch = evtloop["Dispatch"]
            while pycall(Pending, Bool)
                pycall(Dispatch, PyObject)
            end
            pydecref(ea) # deactivate event loop
            pycall(app["ProcessIdle"], PyObject)
        end
    end
    install_doevent(doevent, sec)
end

# Tk: (Tkinter/tkinter module)
function Tk_eventloop(sec::Real=50e-3)
    Tk = pyimport(tkinter_name())
    @doevent begin
        root = Tk["_default_root"]
        if root.o != pynothing
            pycall(root["update"], PyObject)
        end
    end
    install_doevent(doevent, sec)
end
# cache running event loops (so that we don't start any more than once)
if VERSION < v"0.4.0-dev+5322"
    const eventloops = Dict{Symbol,Compat.Timer2}()
else
    const eventloops = Dict{Symbol,Timer}()
end

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
        elseif gui == :qt_pyside
            eventloops[gui] = qt_eventloop("PySide", sec)
        elseif gui == :qt
            try
                eventloops[gui] = qt_eventloop("PyQt4", sec)
            catch
                eventloops[gui] = qt_eventloop("PySide", sec)
            end
        else
            throw(ArgumentError("unsupported GUI type $gui"))
        end
    end
    gui
end

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
