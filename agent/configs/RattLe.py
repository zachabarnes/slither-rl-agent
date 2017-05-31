class config():
    # env config

    # Render during train phase? prob no
    render_train     = False
    # Render during test phase? also prob no
    render_test      = False
    # Current env, (deprecated) (create slither env assumes this)
    env_name         = "internet.SlitherIO-v0"
    # Use custom render over usual render (can be good for visualizing preprocessing)
    overwrite_render = True
    # Record videos?
    record           = True
    #?? pix value?
    high             = 255.

    # output config
    output_path  = "results/RattLe/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    # Number of episodes used in evaluating a current model
    num_episodes_test = 3
    grad_clip         = True
    clip_val          = 10
    #How many train steps before save
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params

    nsteps_train       = 500000
    batch_size         = 32
    buffer_size        = 100
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 1
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 500000
    learning_start     = 500
