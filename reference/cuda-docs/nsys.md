<!-- source: https://docs.nvidia.com/nsight-systems/UserGuide/index.html -->
# User Guide

NVIDIA Nsight Systems user guide

## Preparing Your Application for Profiling

Nsight Systems does not require any application changes to enable profiling; however, by making some simple modifications and additions, you can greatly increase the effectiveness of your profiling and the usability of the resulting data.

### Focused Profiling

By default, Nsight Systems collects a profile over the entire run of your application. But, as explained below, you typically only want to profile the region(s) of your application containing some or all of the performance-critical code. Limiting profiling to performance-critical regions reduces the amount of data that both you and the tools must process, and focuses attention on the code where optimization will result in the greatest performance gains.

There are several common situations where limiting profiling to a region of the application is helpful.

- The application is a test harness wrapping all or part of your algorithm. The test harness may initial the data, run the algorithm cold, and then check the results for correctness. Using a test harness is a common and productive way to quickly iterate and test algorithm changes. When profiling, you will want to collect profile data for the functionality, but not for the test harness initialization and validation.

- The application operates in phases, where a different set of algorithms is active in each phase. When the performance of each phase of the application can be optimized independently of the others, you want to profile each phase separately to focus your optimization efforts.

- The application contains algorithms that operate over a large number of iterations, but the performance of the algorithm does not vary significantly across those iterations. In this case you can collect profile data from a subset of the iterations.

Nsight Systems supports two methods of code annotations to limit profile duration.

- To limit profiling to a region of your CUDA application, CUDA provides functions to start and stop data collection. cudaProfilerStart() is used to start profiling and cudaProfilerStop() is used to stop profiling. To use these functions you must include cuda_profiler_api.h.

- To limit profiling to a region of CPU activity, you can use the NVIDIA Tools Extension API (NVTX) to set range(s) for profiling.

### Marking and Labeling Regions

To understand what the application’s CPU threads are doing beyond CUDA function calls, you can use the [NVIDIA Tools Extension API (NVTX)](https://github.com/NVIDIA/NVTX). When you add NVTX markers and ranges to your application, the Timeline View shows when your CPU threads are executing within those regions. The Timeline View also projects these ranges onto the GPU timeline, allowing you to see what GPU activity was launched within each CPU range.

Using custom names for CPU and CUDA resources can also improve understanding of application behavior, especially for applications that have many host threads, devices, contexts, or streams. You can use the NVIDIA Tools Extension API to assign custom names for your CPU and GPU resources. Your custom names will then be displayed in the Timeline View.

## Profiling from the CLI

### Installing the CLI on Your Target

The Nsight Systems CLI provides a simple interface to collect on a target without using the GUI. The collected data can then be copied to any system and analyzed later.

The CLI is distributed in the Target directory of the standard Nsight Systems download package.

If you wish to run the CLI without root (recommended mode), you will want to install in a directory where you have full access.

Note

You must run the CLI on Windows as administrator.

### Command Line Options

The Nsight Systems command lines can have one of two forms:

``` cpp
nsys [global_option]
```

or

``` cpp
nsys [command_switch][optional command_switch_options][application] [optional application_options]
```

All command line options are case-sensitive. For command switch options, when short options are used, the parameters should follow the switch after a space; e.g., `-s`` ``process-tree`. When long options are used, the switch should be followed by an equal sign and then the parameter(s); e.g., `--sample=process-tree`.

For this version of Nsight Systems, if you launch a process from the command line to begin analysis, the launched process will be terminated when collection is complete, including runs with `--duration` set, unless the user specifies the `--kill`` ``none` option (details below). The exception is that if the user uses NVTX, cudaProfilerStart/Stop, or hotkeys to control the duration, the application will continue unless `--kill` is set.

The Nsight Systems CLI supports concurrent analysis by using sessions. Each Nsight Systems session is defined by a sequence of CLI commands that define one or more collections (e.g., when and what data is collected). A session begins with either a start, launch, or profile command. A session ends with a shutdown command, when a profile command terminates, or, if requested, when all the process tree(s) launched in the session exit. Multiple sessions can run concurrently on the same system.

#### CLI Global Options

| Short | Long | Description |
|----|----|----|
| -h | `--help` | Help message providing information about available command switches and their options. |
| -v | `--version` | Output Nsight Systems CLI version information. |

### CLI Command Switches

The Nsight Systems command line interface can be used in two modes. You may launch your application and begin analysis with options specified to the `nsys`` ``profile` command. Alternatively, you can control the launch of an application and data collection using interactive CLI commands.

| Command | Description |
|----|----|
| analyze | Post process existing Nsight Systems result, either in .nsys-rep or SQLite format, to generate expert systems report. |
| export | Generates an export file from an existing `.nsys-rep` file. For more information about the exported formats see the `/documentation/nsys-exporter` directory in your Nsight Systems installation directory. |
| finalize | Generate report files from deferred collections. When `nsys`` ``stop`` ``--defer-report` is used, the stop command returns quickly and raw data is retained without generating a report. The `finalize` command processes these deferred collections and produces `.nsys-rep` files. |
| launch | In interactive mode, launches an application in an environment that supports the requested options. The launch command can be executed before or after a start command. |
| profile | A fully formed profiling description requiring and accepting no further input. The command switch options used (see below table) determine when the collection starts, stops, what collectors are used (e.g., API trace, IP sampling, etc.), what processes are monitored, etc. |
| recipe | Post process one or more existing Nsight Systems results to generate statistical information and create various plots. See the **Post-Collection Analysis Guide** for details. |
| sessions | Gives information about all sessions running on the system. |
| shutdown | Disconnects the CLI process from the launched application and forces the CLI process to exit. If a collection is pending or active, it is canceled. |
| start | Starts a collection in interactive mode. The start command can be executed before or after a launch command. |
| stats | Post process existing Nsight Systems result, either in `.nsys-rep` or SQLite format, to generate statistical information. |
| status | Reports on the status of a CLI-based collection or the suitability of the profiling environment. |
| stop | Stops a collection that was started in interactive mode. When executed, all active collections stop, the CLI process terminates but the application continues running. |

#### CLI Profile Command Switch Options

After choosing the `profile` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] profile [options] [application] [application-arguments]
```

[TABLE]

#### CLI Analyze Command Switch Options

The `nsys`` ``analyze` command generates and outputs a report to the terminal using expert system rules on existing results. Reports are generated from an SQLite export of a .nsys-rep file. If a .nsys-rep file is specified, Nsight Systems will look for an accompanying SQLite file and use it. If no SQLite export file exists, one will be created.

After choosing the `analyze` command switch, the following options are available. Usage:

`nsys`` ``[global-options]`` ``analyze`` ``[options]`` ``[input-file]`

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--help` | \<tag\>, `none` | Print the help message. The option can take one optional argument that will be used as a tag. If a tag is provided, only options relevant to the tag will be printed. |
| `--format` or `-f` | column, table, csv, tsv, json, hdoc, htable, . | Specify the output format. The special name “.” indicates the default format for the given output. The default format for console is column, while files and process outputs default to csv. This option may be used multiple times. Multiple formats may also be specified using a comma-separated list (\<name\[:args…\]\[,name\[:args…\]…\]\>). See options available with each format at [Available Export Formats](../AnalysisGuide/index.html#available-export-formats) . |
| `--force-export` | true, `false` | Force a re-export of the SQLite file from the specified report, even if an SQLite file already exists. |
| `--force-overwrite` | true, `false` | Overwrite any existing output files. |
| `--help-formats` | \<format_name\>, ALL, `[none]` | With no argument, list a summary of the available output formats. If a format name is given, a more detailed explanation of the format is displayed. If `ALL` is given, a more detailed explanation of all available formats is displayed. |
| `--help-rules` | \<rule_name\>, ALL, `[none]` | With no argument, list available rules with a short description. If a rule name is given, a more detailed explanation of the rule is displayed. If `ALL` is given, a more detailed explanation of all available rules is displayed. |
| `--output` or `-o` | `-`, @\<command\>, \<basename\>, . | Specify the output mechanism. There are three output mechanisms: print to console, output to file, or output to command. This option may be used multiple times. Multiple outputs may also be specified using a comma-separated list. If the given output name is “-”, the output will be displayed on the console. If the output name starts with “@”, the output designates a command to run. The nsys command will be executed and the analysis output will be piped into the command. Any other output is assumed to be the base path and name for a file. If a file basename is given, the filename used will be: \<basename\>\_\<analysis&args\>.\<output_format\>. The default base (including path) is the name of the SQLite file (as derived from the input file or `--sqlite` option), minus the extension. The output “.” can be used to indicate the analysis should be output to a file, and the default basename should be used. To write one or more analysis outputs to files using the default basename, use `--output`. If the output starts with “@”, the nsys command output is piped to the given command. The command is run, and the output is piped to the command’s stdin (standard-input). The command’s stdout and stderr remain attached to the console, so any output will be displayed directly to the console. Be aware there are some limitations in how the command string is parsed. No shell expansions (including \*, ?, \[\], and ~) are supported. The command cannot be piped to another command, nor redirected to a file using shell syntax. The command and command arguments are split on whitespace, and no quotes (within the command syntax) are supported. For commands that require complex command line syntax, it is suggested that the command be put into a shell script file, and the script designated as the output command. |
| `--quiet` or `-q` |  | Do not display verbose messages, only display errors. |
| `--rule` or `-r` | cuda_memcpy_async, cuda_memcpy_sync, cuda_memset_sync, cuda_api_sync, gpu_gaps, gpu_time_util, dx12_mem_ops, `all` | Specify the rule(s) to execute, including any arguments. This option may be used multiple times. Multiple rules may also be specified using a comma-separated list. See [Expert Systems Analysis](../AnalysisGuide/index.html#expert-systems-analysis) section and `--help-rules` switch for details on all rules. |
| `--sqlite` | \<file.sqlite\> | Specify the SQLite export filename. If this file exists, it will be used. If this file doesn’t exist (or if `--force-export` was given) this file will be created from the specified .nsys-rep file before processing. This option cannot be used if the specified input file is also an SQLite file. |
| `--timeunit` | nsec, usec, msec, `nanoseconds`, microseconds, milliseconds, seconds | Set basic unit of time. The argument of the switch is matched by using the longest prefix matching. This means that it is not necessary to write a whole word as the switch argument. It is similar to passing a “:time=\<unit\>” argument to every formatter, although the formatter uses more strict naming conventions. See `nsys`` ``analyze`` ``--help-formats`` ``column` for detailed information on unit conversion. |

#### CLI Export Command Switch Options

After choosing the `export` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] export [options] [nsys-rep-file]
```

[TABLE]

#### CLI Finalize Command Switch Options

After choosing the `finalize` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] finalize [options] [subcommand]
```

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--discard` | all, \<uuid\> | Discard deferred collections without generating reports. Use `all` to discard all deferred collections, or specify a UUID to discard a specific one. Use `nsys`` ``finalize`` ``list` to see available UUIDs. |
| `--force-overwrite` or `-f` | true, **false** | If true, overwrite existing report files when finalizing. Default is false. |
| `--help` | \<tag\> | Print the help message. The option can take one optional argument that will be used as a tag. If a tag is provided, only options relevant to the tag will be printed. |
| `--id` | \<uuid\> | Finalize a specific deferred collection identified by its UUID. Use `nsys`` ``finalize`` ``list` to see available UUIDs. |
| `--output` or `-o` | directory path | Output directory for finalized report files. The original filename is retained. By default, reports are placed in the directory specified during collection. |
| `--session` | session name | Filter deferred collections by session name. |

##### CLI Finalize List Subcommand

The `finalize`` ``list` subcommand displays pending deferred collections. Usage:

``` cpp
nsys [global-options] finalize list [options]
```

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--output-format` or `-f` | **plain**, json | Controls the output format. |
| `--session` | session name | Filter deferred collections by session name. |
| `--show-header` or `-p` | **true**, false | Controls whether a header should appear in the output. |

#### CLI Launch Command Switch Options

After choosing the `launch` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] launch [options] <application> [application-arguments]
```

[TABLE]

#### CLI Sessions Command Switch Subcommands

After choosing the `sessions` command switch, the following subcommands are available. Usage:

``` cpp
nsys [global-options] sessions [subcommand]
```

| Subcommand | Description |
|----|----|
| list | List all active sessions including ID, name, and state information |

##### CLI Sessions List Command Switch Options

After choosing the `sessions`` ``list` command switch, the following options are available. Usage:

``` text
nsys [global-options] sessions list [options]
```

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--help` | \<tag\> | Print the help message. The option can take one optional argument that will be used as a tag. If a tag is provided, only options relevant to the tag will be printed. |
| `--show-header` or `-p` | **true**, false | Controls whether a header should appear in the output. |
| `--output-format` or `-f` | **plain**, json | Output format used for session list. |

#### CLI Shutdown Command Switch Options

After choosing the `shutdown` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] shutdown [options]
```

[TABLE]

#### CLI Start Command Switch Options

After choosing the `start` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] start [options]
```

[TABLE]

#### CLI Stats Command Switch Options

The `nsys`` ``stats` command generates a series of summary or trace reports. These reports can be output to the console, or to individual files, or piped to external processes. Reports can be rendered in a variety of different output formats, from human readable columns of text, to formats more appropriate for data exchange, such as CSV.

Reports are generated from an SQLite export of a .nsys-rep file. If a .nsys-rep file is specified, Nsight Systems will look for an accompanying SQLite file and use it. If no SQLite file exists, one will be exported and created.

Individual reports are generated by calling out to scripts that read data from the SQLite file and return their report data in CSV format. Nsight Systems ingests this data and formats it as requested, then displays the data to the console, writes it to a file, or pipes it to an external process. Adding new reports is as simple as writing a script that can read the SQLite file and generate the required CSV output. See the shipped scripts as an example. Both reports and formatters may take arguments to tweak their processing. For details on shipped scripts and formatters, see [Statistical Analysis](../AnalysisGuide/index.html#statistical-analysis) .

Reports are processed using a three-tuple that consists of:

1.  The requested report (and any arguments),

2.  The presentation format (and any arguments), and

3.  The output (filename, console, or external process).

The first report specified uses the first format specified, and is presented via the first output specified. The second report uses the second format for the second output, and so forth. If more reports are specified than formats or outputs, the format and/or output list is expanded to match the number of provided reports by repeating the last specified element of the list (or the default, if nothing was specified).

`nsys`` ``stats` is a very powerful command and can handle complex argument structures, please see the topic below on Example Stats Command Sequences.

After choosing the `stats` command switch, the following options are available. Usage:

`nsys`` ``[global-options]`` ``stats`` ``[options]`` ``[input-file]`

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--help` | \<tag\> | Print the help message. The option can take one optional argument that will be used as a tag. If a tag is provided, only options relevant to the tag will be printed. |
| `--format` or `-f` | column, table, csv, tsv, json, hdoc, htable, . | Specify the output format. The special name “.” indicates the default format for the given output. The default format for console is column, while files and process outputs default to csv. This option may be used multiple times. Multiple formats may also be specified using a comma-separated list (`<name[:args...][,name[:args...]...]>`). See [Statistical Reports Shipped With Nsight Systems](../AnalysisGuide/index.html#available-statistical-reports) for options available with each report. |
| `--force-export` | true, **false** | Force a re-export of the SQLite file from the specified .nsys-rep file, even if an SQLite file already exists. |
| `--force-overwrite` | true, **false** | Overwrite any existing report file(s). |
| `--help-formats` | \<format_name\>, ALL, **\[none\]** | With no argument, give a summary of the available output formats. If a format name is given, a more detailed explanation of that format is displayed. If `ALL` is given, a more detailed explanation of all available formats is displayed. |
| `--help-reports` | \<report_name\>, ALL, **\[none\]** | With no argument, list a summary of the available summary and trace reports. If a report name is given, a more detailed explanation of the report is displayed. If `ALL` is given, a more detailed explanation of all available reports is displayed. |
| `--output` or `-o` | **-**, @\<command\>, \<basename\>, . | Specify the output mechanism. There are three output mechanisms: print to console, output to file, or output to command. This option may be used multiple times. Multiple outputs may also be specified using a comma-separated list. If the given output name is “-”, the output will be displayed on the console. If the output name starts with “@”, the output designates a command to run. The nsys command will be executed and the analysis output will be piped into the command. Any other output is assumed to be the base path and name for a file. If a file basename is given, the filename used will be: `<basename>\_<analysis&args>.<output_format>`. The default base (including path) is the name of the SQLite file (as derived from the input file or `--sqlite` option), minus the extension. The output “.” can be used to indicate the analysis should be output to a file, and the default basename should be used. To write one or more analysis outputs to files using the default basename, use the option: `--output`. If the output starts with “@”, the nsys command output is piped to the given command. The command is run, and the output is piped to the command’s stdin (standard-input). The command’s stdout and stderr remain attached to the console, so any output will be displayed directly to the console. Be aware there are some limitations in how the command string is parsed. No shell expansions (including \*, ?, \[\], and ~) are supported. The command cannot be piped to another command, nor redirected to a file using shell syntax. The command and command arguments are split on whitespace, and no quotes (within the command syntax) are supported. For commands that require complex command line syntax, it is suggested that the command be put in a shell script file, and that be designated as the output command. |
| `--quiet` or `-q` |  | Do not display verbose messages, only display errors. |
| `--report` or `-r` | See **Report Scripts** | Specify the report(s) to generate, including any arguments. This option may be used multiple times. Multiple reports may also be specified using a comma-separated list (`<name[:args...]`` ``[,name[:args...]...]>`). If no reports are specified, the following will be used as the default report set: nvtx_sum, osrt_sum, cuda_api_sum, cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, openmp_sum, opengl_khr_range_sum, opengl_khr_gpu_range_sum, vulkan_marker_sum, vulkan_gpu_marker_sum, dx11_pix_sum, dx12_gpu_marker_sum, dx12_pix_sum, wddm_queue_sum, um_sum, um_total_sum, um_cpu_page_faults_sum, openacc_sum. See [Report Scripts](index.html#report-scripts) for details about existing built-in scripts and how to make your own. |
| `--report-dir` | \<path\> | Add a directory to the path used to find report scripts. This is usually only needed if you have one or more directories with personal scripts. This option may be used multiple times. Each use adds a new directory to the end of the path. A search path can also be defined using the environment variable `NSYS_STATS_REPORT_PATH`. Directories added this way will be added after the application flags. The last two entries in the path will always be the current working directory, followed by the directory containing the shipped `nsys` reports. |
| `--sqlite` | \<file.sqlite\> | Specify the SQLite export filename. If this file exists, it will be used. If this file doesn’t exist (or if `--force-export` was given) this file will be created from the specified .nsys-rep file before processing. This option cannot be used if the specified input file is also an SQLite file. |
| `--timeunit` | nsec, usec, msec, **nanoseconds**, microseconds, milliseconds, seconds | Set basic unit of time. The argument of the switch is matched by using the longest prefix matching, meaning that it is not necessary to write a whole word as the switch argument. It is similar to passing a `:time=<unit>` argument to every formatter, although the formatter uses more strict naming conventions. See `nsys`` ``stats`` ``--help-formats`` ``column` for more detailed information on unit conversion. |

#### CLI Status Command Switch Options

The `nsys`` ``status` command returns the current state of the CLI. After choosing the `status` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] status [options]
```

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--all` |  | Prints information for all the available profiling environments. |
| `--environment` or `-e` |  | Returns information about the system regarding suitability of the profiling environment. |
| `--help` | \<tag\> | Print the help message. The option can take one optional argument that will be used as a tag. If a tag is provided, only options relevant to the tag will be printed. |
| `--network` or `-n` |  | Returns information about the system regarding suitability of the network profiling environment. |
| `--session` | session identifier | Print the status of the indicated session. The option argument must represent a valid session name or ID as reported by `nsys`` ``sessions`` ``list`. Any `%q{ENV_VAR}` pattern will be substituted with the value of the environment variable. Any `%h` pattern will be substituted with the hostname of the system. Any `%%` pattern will be substituted with `%`. |

#### CLI Stop Command Switch Options

After choosing the `stop` command switch, the following options are available. Usage:

``` cpp
nsys [global-options] stop [options]
```

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--defer-report` |  | Defer the generation of the report file so that the stop command returns quickly. The raw data will be retained in a temporary location. Use `nsys`` ``finalize` to generate the report file later. This is useful when performing multiple start/stop cycles on a long-running application. |
| `--help` | \<tag\> | Print the help message. The option can take one optional argument that will be used as a tag. If a tag is provided, only options relevant to the tag will be printed. |
| `--keep` | time in seconds | Indicate how many seconds of collected data previous to the stop command should be retained in the result file. Zero is treated as a special setting that retains all of the data. |
| `--session` | session identifier | Stop the indicated session. The option argument must represent a valid session name or ID as reported by `nsys`` ``sessions`` ``list`. Any `%q{ENV_VAR}` pattern will be substituted with the value of the environment variable. Any `%h` pattern will be substituted with the hostname of the system. Any `%%` pattern will be substituted with `%`. |

### Example Single Command Lines

**Version Information**

``` text
nsys -v
```

Effect: Prints tool version information to the screen.

**Run with elevated privilege**

``` text
sudo nsys profile <app>
```

Effect: Nsight Systems CLI (and target application) will run with elevated privilege. This is necessary for some features, such as FTrace or system-wide CPU sampling. If you don’t want the target application to be elevated, use `--run-as` option.

**Default analysis run**

``` text
nsys profile <application>
    [application-arguments]
```

Effect: Launch the application using the given arguments. Start collecting immediately and end collection when the application stops. Trace CUDA, OpenGL, NVTX, and OS runtime libraries APIs. Collect CPU Instruction Pointer (IP) sampling information and thread scheduling information. With Nsight Systems Embedded Platforms Edition this will only analysis the single process. With Nsight Systems Workstation Edition this will trace the process tree. Generate the report#.nsys-rep file in the default location, incrementing the report number if needed to avoid overwriting any existing output files.

**Limited trace only run**

``` text
nsys profile --trace=cuda,nvtx -d 20
    --sample=none --cpuctxsw=none -o my_test <application>
    [application-arguments]
```

Effect: Launch the application using the given arguments. Start collecting immediately and end collection after 20 seconds or when the application ends. Trace CUDA and NVTX APIs. Do not collect CPU sampling information or thread scheduling information. Profile any child processes. Generate the output file as `my_test.nsys-rep` in the current working directory.

**Delayed start run**

``` text
nsys profile -e TEST_ONLY=0 -y 20
    <application> [application-arguments]
```

Effect: Set environment variable TEST_ONLY=0. Launch the application using the given arguments. Start collecting after 20 seconds and end collection at application exit. Trace CUDA, OpenGL, NVTX, and OS runtime libraries APIs. Collect CPU sampling and thread schedule information. Profile any child processes. Generate the report#.nsys-rep file in the default location, incrementing if needed to avoid overwriting any existing output files.

**Run application, start/stop collection using NVTX**

``` text
nsys profile -c nvtx -w true -p MESSAGE@DOMAIN <application> [application-arguments]
```

Effect: Create interactive CLI process and set it up to begin collecting as soon as an NVTX range with a given message in a given domain (capture range) is opened. Launch application for default analysis, sending application output to the terminal. Stop collection when all capture ranges are closed, when the user calls `nsys`` ``stop`, or when the root process terminates. Generate the `report#.nsys-rep` in the default location.

Note

The Nsight Systems CLI only triggers the profiling session for the first capture range.

NVTX capture range can be specified:

- Message@Domain: All ranges with given message in given domain are capture ranges. For example:

  ``` text
  nsys profile -c nvtx -w true -p profiler@service ./app
  ```

  This would make the profiling start when the first range with message “profiler” is opened in domain “service.”

- Message@\*: All ranges with given message in all domains are capture ranges. For example:

  ``` text
  nsys profile -c nvtx -w true -p profiler@* ./app
  ```

  This would make the profiling start when the first range with message “profiler” is opened in any domain.

- Message: All ranges with given message in default domain are capture ranges. For example:

  ``` text
  nsys profile -c nvtx -w true -p profiler ./app
  ```

  This would make the profiling start when the first range with message “profiler” is opened in the default domain.

- By default, only messages provided by NVTX registered strings are considered. This avoids the need to perform a string match on every NVTX string encountered in the application, which creates significant additional overhead. It is strongly recommended to always use NVTX registered strings. If you do not use registered strings you will have to enable the full match by launching your application with `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` environment:

  ``` text
  nsys profile -c nvtx -w true -p profiler@service -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 ./app
  ```

Note

The separator ‘@’ can be escaped with backslash ‘\\. If multiple separators without escape character are specified, only the last one is applied, all others are discarded.

**Collect ftrace events**

``` text
nsys profile --ftrace=drm/drm_vblank_event
    -d 20
```

Effect: Collect ftrace `drm_vblank_event` events for 20 seconds. Generate the report#.nsys-rep file in the current working directory. Note that ftrace event collection requires running as root. To get a list of ftrace events available from the kernel, run the following:

``` text
sudo cat /sys/kernel/debug/tracing/available_events
```

**Run GPU metric sampling on one TU10x**

``` text
nsys profile --gpu-metrics-devices=0
    --gpu-metrics-set=tu10x-gfxt <application>
```

Effect: Launch application. Collect default options and GPU metrics for the first GPU (a TU10x), using the tu10x-gfxt metric set at the default frequency (10 kHz). Profile any child processes. Generate the `report#.nsys-rep` file in the default location, incrementing if needed to avoid overwriting any existing output files.

**Run GPU metric sampling on all GPUs at a set frequency**

``` text
nsys profile --gpu-metrics-devices=all
    --gpu-metrics-frequency=20000 <application>
```

Effect: Launch application. Collect default options and GPU metrics for all available GPUs using the first suitable metric set for each and sampling at 20 kHz. Profile any child processes. Generate the report#.nsys-rep file in the default location, incrementing if needed to avoid overwriting any existing output files.

**Collect CPU IP/backtrace and CPU context switch**

``` text
nsys profile --sample=system-wide --duration=5
```

Effect: Collects both CPU IP/backtrace samples using the default backtrace mechanism and traces CPU context switch activity for the whole system for 5 seconds. Note that it requires root permission or a Linux paranoid level of 0 or less to run. No hardware or OS events are sampled. Post processing of this collection will take longer due to the large number of symbols to be resolved caused by system-wide sampling.

**Get list of available CPU core events and metrics**

``` text
nsys profile --cpu-metrics=help
```

Effect: Lists the CPU core events and derived metrics that can be sampled and also gives the maximum number of CPU events that can be sampled concurrently.

**Collect system-wide CPU events and metrics, and trace application**

``` text
nsys profile --event-sample=system-wide
    --cpu-metrics=ITLB_WALK,DTLB_WALK,ipc --event-sampling-interval=5 <app> [app args]
```

Effect:Collects CPU IP/backtrace samples using the default backtrace mechanism, traces CPU context switch activity, collects CPU core events: ITLB_WALK, DTLB_WALK and CPU core metrics: ipc every 5 ms for the whole system. Note that it requires root permission or a Linux paranoid level of 0 or less to run. Note that CUDA, NVTX, OpenGL, and OSRT within the app launched by Nsight Systems are traced by default while using this command. Post processing of this collection will take longer due to the large number of symbols to be resolved caused by system-wide sampling.

**Collect custom ETW trace using configuration file**

``` text
nsys profile --etw-provider=file.JSON
```

Effect: Configure custom ETW collectors using the contents of file.JSON. Collect data for 20 seconds. Generate the `report#.nsys-rep` file in the current working directory.

A template JSON configuration file is located at in the Nsight Systems installation directory as `\\target-windows-x64\\etw_providers_template.json`. This path will show up automatically if you call the following:

``` text
nsys profile --help
```

The **level** attribute can only be set to one of the following:

- TRACE_LEVEL_CRITICAL

- TRACE_LEVEL_ERROR

- TRACE_LEVEL_WARNING

- TRACE_LEVEL_INFORMATION

- TRACE_LEVEL_VERBOSE

The **flags** attribute can only be set to one or more of the following:

- EVENT_TRACE_FLAG_ALPC

- EVENT_TRACE_FLAG_CSWITCH

- EVENT_TRACE_FLAG_DBGPRINT

- EVENT_TRACE_FLAG_DISK_FILE_IO

- EVENT_TRACE_FLAG_DISK_IO

- EVENT_TRACE_FLAG_DISK_IO_INIT

- EVENT_TRACE_FLAG_DISPATCHER

- EVENT_TRACE_FLAG_DPC

- EVENT_TRACE_FLAG_DRIVER

- EVENT_TRACE_FLAG_FILE_IO

- EVENT_TRACE_FLAG_FILE_IO_INIT

- EVENT_TRACE_FLAG_IMAGE_LOAD

- EVENT_TRACE_FLAG_INTERRUPT

- EVENT_TRACE_FLAG_JOB

- EVENT_TRACE_FLAG_MEMORY_HARD_FAULTS

- EVENT_TRACE_FLAG_MEMORY_PAGE_FAULTS

- EVENT_TRACE_FLAG_NETWORK_TCPIP

- EVENT_TRACE_FLAG_NO_SYSCONFIG

- EVENT_TRACE_FLAG_PROCESS

- EVENT_TRACE_FLAG_PROCESS_COUNTERS

- EVENT_TRACE_FLAG_PROFILE

- EVENT_TRACE_FLAG_REGISTRY

- EVENT_TRACE_FLAG_SPLIT_IO

- EVENT_TRACE_FLAG_SYSTEMCALL

- EVENT_TRACE_FLAG_THREAD

- EVENT_TRACE_FLAG_VAMAP

- EVENT_TRACE_FLAG_VIRTUAL_ALLOC

**Typical case: profile a Python script that uses CUDA**

``` text
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx
    --cudabacktrace=all --python-backtrace=cuda --python-sampling=true
    --delay=60 python my_dnn_script.py
```

Effect: Launch a Python script and start profiling it 60 seconds after the launch, tracing CUDA, cuDNN, cuBLAS, OS runtime APIs, and NVTX as well as collecting CPU IP and Python call stack samples and thread scheduling information. CUDA and Python call stacks are also collected on CUDA API calls.

**Typical case: profile a Python script that uses PyTorch and CUDA**

``` text
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace-shapes,autograd-nvtx
    --cudabacktrace=all --python-backtrace=cuda --python-sampling=true
    --delay=60 python my_torch_script.py
```

Effect: Launch a Python script and start profiling it 60 seconds after the launch, tracing CUDA, cuDNN, cuBLAS, OS runtime APIs, and NVTX as well as collecting CPU IP and Python call stack samples and thread scheduling information. PyTorch functions are traced, and tensor shapes are collected via `--pytorch=functions-trace-shapes` to provide detailed information about the structure and execution of the neural network model. CUDA and Python call stacks are also collected on CUDA API calls.

**Typical case: profile an app that uses Vulkan**

``` text
nsys profile --trace=vulkan,osrt,nvtx
    --delay=60 ./myapp
```

Effect: Launch an app and start profiling it 60 seconds after the launch, tracing Vulkan, OS runtime APIs, and NVTX as well as collecting CPU sampling and thread schedule information.

### Example Interactive CLI Command Sequences

**Collect from beginning of application, end manually**

``` text
nsys start --stop-on-exit=false
nsys launch --trace=cuda,nvtx --sample=none <application> [application-arguments]
nsys stop
```

Effect: Create interactive CLI process and set it up to begin collecting as soon as an application is launched. Launch the application, set up to allow tracing of CUDA and NVTX as well as collection of thread schedule information. Stop only when explicitly requested. Generate the report#.nsys-rep in the default location.

Note

If you start a collection and fail to stop the collection (or if you are allowing it to stop on exit, and the application runs for too long), your system’s storage space may be filled with collected data causing significant issues for the system. Nsight Systems will collect a different amount of data/sec depending on options, but in general Nsight Systems does not support runs of more than 5 minutes duration.

**Run application, begin collection manually, run until process ends**

``` text
nsys launch -w true <application> [application-arguments]
nsys start
```

Effect: Create interactive CLI and launch an application set up for default analysis. Send application output to the terminal. No data is collected until you manually start collection at area of interest. Profile until the application ends. Generate the report#.nsys-rep in the default location.

Note

If you launch an application and that application and any descendants exit before start is called, Nsight Systems will create a fully formed .nsys-rep file containing no data.

**Run application, name the session, keep only the last seconds**

``` text
nsys start --session-new=mysession
nsys launch --session=mysession myapp [application-arguments]
nsys stop --session=mysession --keep=3
```

Effect: Create named interactive CLI process and launch your app with default collection options. Manually stop that session and keep only the last three seconds of data.

Note

Currently Nsight Systems will collect all the data and then trim the data at stop time. In the future we will add an option that does the collection in a ring buffer, so that if the user knows ahead of time how many seconds of data they wish to save we can avoid using unneeded memory.

**Run application, start/stop collection using cudaProfilerStart/Stop**

``` text
nsys start -c cudaProfilerApi
nsys launch -w true <application> [application-arguments]
```

Effect: Create interactive CLI process and set it up to begin collecting as soon as a `cudaProfileStart()` is detected. Launch application for default analysis, sending application output to the terminal. Stop collection at next call to `cudaProfilerStop`, when the user calls `nsys`` ``stop`, or when the root process terminates. Generate the `report#.nsys-rep` in the default location.

Note

If you call `nsys`` ``launch` before `nsys`` ``start`` ``-c`` ``cudaProfilerApi` and the code contains a large number of short duration cudaProfilerStart/Stop pairs, Nsight Systems may be unable to process them correctly, causing a fault. This will be corrected in a future version.

Note

Use the Nsight Systems CLI option `--capture-range-end-repeat` to capture a separate report file for each capture range defined by calls to cudaProfilerStart/Stop. To avoid overwriting report files unexpectedly, Nsight Systems will ignore the `--force-overwrite` option in this case.

**Run application, start/stop collection using NVTX**

``` text
nsys start -c nvtx
nsys launch -w true -p MESSAGE@DOMAIN <application> [application-arguments]
```

Effect: Create interactive CLI process and set it up to begin collecting as soon as an NVTX range with a given message in a given domain (capture range) is opened. Launch application for default analysis, sending application output to the terminal. Stop collection when all capture ranges are closed, when the user calls `nsys`` ``stop`, or when the root process terminates. Generate the `report#.nsys-rep` in the default location.

Note

The Nsight Systems CLI only triggers the profiling session for the first capture range.

NVTX capture range can be specified:

- Message@Domain: All ranges with given message in given domain are capture ranges. For example:

  ``` text
  nsys launch -w true -p profiler@service ./app
  ```

  This would make the profiling start when the first range with message “profiler” is opened in domain “service.”

- Message@\*: All ranges with given message in all domains are capture ranges. For example:

  ``` text
  nsys launch -w true -p profiler@* ./app
  ```

  This would make the profiling start when the first range with message “profiler” is opened in any domain.

- Message: All ranges with given message in default domain are capture ranges. For example:

  ``` text
  nsys launch -w true -p profiler ./app
  ```

  This would make the profiling start when the first range with message “profiler” is opened in the default domain.

- By default, only messages provided by NVTX registered strings are considered. This avoids the need to perform a string match on every NVTX string encountered in the application, which creates significant additional overhead. It is strongly recommended to always use NVTX registered strings. If you do not use registered strings you will have to enable the full match by launching your application with `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` environment:

  ``` text
  nsys launch -w true -p profiler@service -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 ./app
  ```

Note

The separator ‘@’ can be escaped with backslash ‘\\. If multiple separators without escape character are specified, only the last one is applied, all others are discarded.

**Run application, start/stop collection multiple times**

The interactive CLI supports multiple sequential collections per launch.

``` text
nsys launch <application> [application-arguments]
nsys start
nsys stop
nsys start
nsys stop
nsys shutdown --kill sigkill
```

Effect: Create interactive CLI and launch an application set up for default analysis. Send application output to the terminal. No data is collected until the start command is executed. Collect data from start until stop requested, generate `report#.qstrm` in the current working directory. Collect data from second start until the second stop request, generate `report#.nsys-rep` (incremented by one) in the current working directory. Shutdown the interactive CLI and send sigkill to the target application’s process group.

**Collect multiple regions of interest, defer report generation**

``` text
nsys launch --trace=cuda,nvtx <application> [application-arguments]
nsys start -o region_1
nsys stop --defer-report
nsys start -o region_2
nsys stop --defer-report
nsys start -o region_3
nsys stop --defer-report
nsys shutdown
nsys finalize list
nsys finalize
```

Effect: Launch a long-running application and collect multiple regions of interest. Each `stop`` ``--defer-report` returns quickly because report generation is deferred, allowing the next `start` to begin without delay. After all collections are complete, use `nsys`` ``finalize`` ``list` to see the pending deferred collections and `nsys`` ``finalize` to generate the `.nsys-rep` report files. Output name substitution patterns (`%h`, `%q{}`, `%p`, `%n`, `%%`) are supported.

To finalize a specific collection, pass its UUID from the list output:

``` text
nsys finalize --id <uuid-from-list>
```

To discard deferred collections without generating reports:

``` text
nsys finalize --discard=all
nsys finalize --discard=<uuid>
```

### Example Stats Command Sequences

**Display default statistics**

`nsys`` ``stats`` ``report1.nsys-rep`

Effect: Export an SQLite file named report1.sqlite from report1.nsys-rep (assuming it does not already exist). Print the default reports in column format to the console.

Note: The following two command sequences should present very similar information:

``` text
nsys profile --stats=true <application>
```

or

``` text
nsys profile <application>
```

``` text
nsys stats report1.nsys-rep
```

**Display specific data from a report**

``` text
nsys stats --report cuda_gpu_trace report1.nsys-rep
```

Effect: Export an SQLite file named `report1.sqlite` from `report1.nsys-rep` (assuming it does not already exist). Print the report generated by the `cuda_gpu_trace` script to the console in column format.

**Generate multiple reports, in multiple formats, output multiple places**

``` text
nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --report cuda_api_sum --format csv,column --output .,- report1.nsys-rep
```

Effect: Export an SQLite file named `report1.sqlite` from `report1.nsys-rep` (assuming it does not already exist). Generate three reports. The first, the `cuda_gpu_trace` report, will be output to the file `report1_cuda_gpu_trace.csv` in CSV format. The other two reports, `cuda_gpu_kern_sum` and `cuda_api_sum`, will be output to the console as columns of data. Although three reports were given, only two formats and outputs are given. To reconcile this, both the list of formats and outputs is expanded to match the list of reports by repeating the last element.

**Submit report data to a command**

``` text
nsys stats --report cuda_api_sum --format table \ --output @“grep -E (-|Name|cudaFree” test.sqlite
```

Effect: Open test.sqlite and run the `cuda_api_sum` script on that file. Generate table data and feed that into the command `grep`` ``-E`` ``(-|Name|cudaFree)`. The grep command will filter out everything but the header, formatting, and the `cudaFree` data, and display the results to the console.

Note

When the output name starts with ‘@,’ it is defined as a command. The command is run, and the output of the report is piped to the command’s stdin (standard-input). The command’s stdout and stderr remain attached to the console, so any output will be displayed directly to the console.

Be aware there are some limitations in how the command string is parsed. No shell expansions (including \*, ?, \[\], and ~) are supported. The command cannot be piped to another command, nor redirected to a file using shell syntax. The command and command arguments are split on whitespace, and no quotes (within the command syntax) are supported. For commands that require complex command line syntax, it is suggested that the command be put into a shell script file, and the script designated as the output command

### Example Output from `--stats` Option

The `nsys`` ``stats` command can be used post analysis to generate specific or personalized reports. For a default fixed set of summary statistics to be automatically generated, you can use the `--stats` option with the `nsys`` ``profile` or `nsys`` ``start` command to generate a fixed set of useful summary statistics.

If your run traces CUDA, these include CUDA API, Kernel, and Memory Operation statistics:



If your run traces OS runtime events or NVTX push-pop ranges:



If your run traces graphics debug markers these include DX11 debug markers, DX12 debug markers, Vulkan debug markers or KHR debug markers:



Recipes for these statistics as well as documentation on how to create your own metrics will be available in a future version of the tool.

### System Wide API Trace on Windows

On Windows, Nsight Systems can trace certain APIs (currently supported: DX11, DX12 and Vulkan) in already-running applications, by way of system-wide API trace from the CLI.

To initiate system-wide API tracing, run the Nsight Systems CLI with the `--trace` option including one or more of the supported APIs, the `--system-wide` option set to `true`, and without specifying a target application. System-wide API tracing may be combined with trace types that are always system-wide such as `--trace=wddm`.

To trace a DX11 or DX12 target application, it must gain the system focus, the user must either click on the application window or use Alt+Tab to select it.

For example, to trace multiple DX12 applications with PIX markers and GPU workload trace, as well as WDDM events for the next 20 seconds, run the command:

``` text
nsys profile --trace=dx12-annotations,wddm --dx12-gpu-workload=individual
--duration=20
```

Then click each of the target applications’ windows to give them focus.

To trace a Vulkan target application, it must be launched after the `nsys`` ``profile` command has been executed.

### Symbol Locations on Linux and QNX

On Linux and QNX, user can specify local directories with symbol/debug files and Debuginfod symbol servers.

**The list of directories with symbol files**

- `--debug-symbols` CLI option, for example:

  ``` text
  --debug-symbols=/lib:/root/symbols
  ```

- Alternatively, `DbgFileSearchPath` config option can be used, for example:

  ``` text
  NSYS_CONFIG_DIRECTIVES='DbgFileSearchPath="/lib:/root/symbols"' nsys profile <app>
  ```

On Linux, the default path is `/usr/lib/debug`. On QNX, there is no default path. The search is recursive.

**Debuginfod symbol servers**

- Nsight Systems automatically queries and downloads missing symbols from Debuginfod servers.

- Official public servers exist for Ubuntu, Debian, Fedora and other distributions.

- [Federated servers for multiple distros](https://sourceware.org/elfutils/Debuginfod.html).

- At least one server URL should be provided by `DEBUGINFOD_URLS` environment variable to enable Debuginfod functionality. `DEBUGINFOD_URLS` contains a list of Debuginfod servers (space separated URLs), example (local and public servers):

  ``` text
  export DEBUGINFOD_URLS="http://localhost:8002 https://debuginfod.ubuntu.com"
  nsys profile <app>
  ```

**Debuginfod cache directory**

- Nsight Systems stores downloaded files in:

  - `$DEBUGINFOD_CACHE_PATH/debuginfod_client` - if `DEBUGINFOD_CACHE_PATH` is set.

  - Otherwise, if `XDG_CACHE_HOME` is set, then `$XDG_CACHE_HOME/debuginfod_client` directory will be used.

  - Otherwise, if neither DEBUGINFOD_CACHE_PATH nor XDG_CACHE_HOME are set, `$HOME/.cache/debuginfod_client/` directory is the default location for downloaded files.

- Nsight Systems uses cached files only if Debuginfod functionality is enabled (`DEBUGINFOD_URLS` environment variable is set).

- Nsight Systems also support reading LLVM cache files (`$HOME/.cache/llvm-debuginfod/`).

Nsight Systems shows the download progress (fetching the files from remote HTTP servers can take a long time, especially for system wide CPU sampling mode):

- The total amount of files and how many files are already downloaded.

- User is able to cancel the download process (Ctrl+C):

  ``` text
  Press Ctrl-C to stop symbol files downloading
  [1/16] Downloaded symbol information for /usr/lib/x86_64-linux-gnu/libnss_files-2.31.so
  ...
  ```

### Opening Command Line Results Files for Visualization

**Open CLI results in GUI**

The CLI will generate an .nsys-rep file after analysis is complete. This file can be opened in any GUI that is the same version or a more recent version.

The opening of really large, multi-gigabyte, .nsys-rep files may take up all of the memory on the host computer and lock up the system. This will be fixed in a later version.

**Importing Windows ETL files**

For Windows targets, ETL files captured with Xperf or the `log.cmd` command supplied with GPUView in the Windows Performance Toolkit can be imported to create reports as if they were captured with Nsight Systems’s “WDDM trace” and “Custom ETW trace” features. Simply choose the .etl file from the Import dialog to convert it to a .nsys-rep file.

### Handling Application Launchers (mpirun, deepspeed, etc)

Nsight Systems has built-in API trace support for various communication APIs, such as MPI, OpenSHMEM, UCX, NCCL and NVSHMEM. To execute respective programs on multiple different machines (compute nodes), usually launchers are used, e.g., `mpirun`/`mpiexec` (MPI), `shmemrun`/`oshrun` (OpenSHMEM), `srun` (SLURM) or `deepspeed`.

In **single-node profiling** sessions, the Nsight Systems CLI can be prefixed before the program (binary) or the launcher. In the latter case, the execution of the launcher will also be profiled and all processes will be recorded into the same report file, e.g for mpirun

> `nsys`` ``profile`` ``[nsys`` ``args]`` ``mpirun`` ``[mpirun`` ``args]`` ``...`

In **multi-node profiling** sessions, the Nsight Systems CLI has to be prefixed before the application, but not before the launcher command, e.g. for mpirun

> `mpirun`` ``[mpirun`` ``args]`` ``nsys`` ``profile`` ``[nsys`` ``args]`` ``...`

You can use `%q{OMPI_COMM_WORLD_RANK}` (Open MPI), `%q{PMI_RANK}` (MPICH), `%q{SLURM_PROCID}` (Slurm) or `%p` in the `-o|--output` option to include the rank or process ID into the report file name.

Warning

An error will occur if several processes want to write to the same report file at the same time.

#### Profile a Single Process or a Subset of Processes

It might be reasonable to profile only a single or a few representative processes of a program run, e.g., to reduce the amount of measurement data.

To achieve this, a wrapper script can be used. The following script called *nsys_profile.sh* uses nsys to profile MPI rank 0 only.

``` cpp
#!/bin/bash

# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
  nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 -t mpi "$@"
else
  "$@"
fi
```

You can change the profiling options accordingly and execute with `mpirun`` ``[mpirun`` ``options]`` ``./nsys_profile.sh`` ``./myapp`` ``[app`` ``options]`. The above code can be easily adapted for OpenSHMEM and SLURM launchers.

Note

If only a subset of MPI ranks is profiled, set the environment variable `NSYS_MPI_STORE_TEAMS_PER_RANK=1` to store all members of custom MPI communicators per MPI rank. Otherwise, the execution might hang or fail with an MPI error.

#### DeepSpeed

DeepSpeed provides a parallel launcher, which launches a Python script on multiple nodes and/or GPUs. For multi-node runs, a simple wrapper script (*nsys_profile.sh*) is required to profile with Nsight Systems.

``` cpp
#! /bin/bash
nsys profile -t cuda,mpi,nvtx,cudnn -o rname.%p python ...
```

This above script has to be used with the –no-python option:

``` cpp
deepspeed --no_python [deepspeed args] ./nsys_profile.sh
```

#### Torchrun/Pytorch

Here is an example of using Nsight Systems to selectively profile GPUs in a multi-gpu system using torchrun.

``` cpp
$ cat run.py

import subprocess
import sys
import os local_rank = int(os.environ["LOCAL_RANK"])

args = sys.argv[1:]
args_string = ' '.join(args)

#Define the command to execute
print(f"Profile local rank {local_rank} only")
if local_rank == 0:
  command = "nsys profile -t cuda,nvtx -o test_run python " + args_string
else:
  command = "python " + args_string

#Run the command
subprocess.run(command, shell=True)

$ torchrun --nnodes=1 --nproc-per-node=8 run.py target_python_script.py
```

#### GPU and NIC metrics collection

If multiple instances of `nsys`` ``profile` are executed concurrently on the same node, and GPU and/or NIC metrics collection is enabled, each process will collect metrics for all available NICs and tries to collect GPU metrics for the specified devices. This can be avoided with a simple bash script similar to the following:

``` cpp
#!/bin/bash

# Use $SLURM_LOCALID with srun.
if [ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]; then
  nsys profile --nic-metrics=lf --gpu-metrics-devices=all "$@"
else
  nsys profile "$@"
fi
```

This above script will collect NIC and GPU metrics only for one rank, the node-local rank 0. Alternatively, if one rank per GPU is used, the GPU metrics devices can be specified based on the node-local rank in a wrapper script as follows:

``` cpp
#!/bin/bash

# Use $SLURM_LOCALID with srun.
nsys profile -e CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK} \
  --gpu-metrics-devices=${OMPI_COMM_WORLD_LOCAL_RANK} "$@"
```

## Profiling from the GUI

The GUI launch script, `nsys-ui`, is located in the host directory in your Nsight Systems installation.

### Profiling Linux Targets from the GUI

#### Connecting to the Target Device

Nsight Systems provides a simple interface to profile on localhost or manage multiple connections to Linux or Windows based devices via SSH. The network connections manager can be launched through the device selection dropdown:

On x86_64:

> 

On Tegra:

> 

The dialog has simple controls that allow adding, removing, and modifying connections:

> 

Warning

**Security notice**: SSH is only used to establish the initial connection to a target device, perform checks, and upload necessary files. The actual profiling commands and data are transferred through a raw, unencrypted socket. Nsight Systems should not be used in a network setup where attacker-in-the-middle attack is possible, or where untrusted parties may have network access to the target device.

While connecting to the target device, you will be prompted to input the user’s password. Note that if you choose to remember the password, it will be stored in plain text in the configuration file on the host. Stored passwords are bound to the public key fingerprint of the remote device.

The **No authentication** option is useful for devices configured for passwordless login using `root` username. To enable such a configuration, edit the file `/etc/ssh/sshd_config` on the target and specify the following option:

``` cpp
PermitRootLogin yes
```

Then set empty password using `passwd` and restart the SSH service with `service`` ``ssh`` ``restart`.

**Open ports**: The Nsight Systems agent requires port 22 and port 45555 to be open for listening. You can confirm that these ports are open with the following command:

``` cpp
sudo firewall-cmd --list-ports --permanent
sudo firewall-cmd --reload
```

To open a port use the following command, skip `--permanent` option to open only for this session:

``` cpp
sudo firewall-cmd --permanent --add-port 45555/tcp
sudo firewall-cmd --reload
```

Likewise, if you are running on a cloud system, you must open port 22 and port 45555 for ingress.

**Kernel Version Number** - To check for the version number of the kernel support of Nsight Systems on a target device, run the following command on the remote device:

``` cpp
cat /proc/quadd/version
```

Minimal supported version is 1.82.

Additionally, presence of Netcat command (`nc`) is required on the target device. For example, on Ubuntu this package can be installed using the following command:

``` cpp
sudo apt-get install netcat-openbsd
```

#### System-Wide Profiling Options

#### Target Sampling Options

Target sampling behavior is somewhat different for Nsight Systems Workstation Edition and Nsight Systems Embedded Platforms Edition.

#### Hotkey Trace Start/Stop

Nsight Systems Workstation Edition can use hotkeys to control profiling. Press the hotkey to start and/or stop a trace session from within the target application’s graphic window. This is useful when tracing games and graphic applications that use fullscreen display. In these scenarios, switching to Nsight Systems’ UI would unnecessarily introduce the window manager’s footprint into the trace. To enable the use of Hotkey, check the Hotkey checkbox in the project settings page:

> 

The default hotkey is F12.

#### Launching Processes

Nsight Systems can launch new processes for profiling on target devices. The profiler ensures that all environment variables are set correctly to successfully collect trace information

> 

The **Edit arguments…** link will open an editor window, where every command line argument is edited on a separate line. This is convenient when arguments contain spaces or quotes.

### Profiling Windows Targets from the GUI

Profiling on Windows devices is similar to the profiling on Linux devices. Please refer to the [Profiling Linux Targets from the GUI](index.html#profiling-linux-targets-from-the-gui) section for the detailed documentation and connection information. The major differences on the platforms are listed below:

#### Remoting to a Windows Based Machine

To perform remote profiling to a target Windows based machines, [install and configure an OpenSSH Server on the target machine.](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse).

#### Hotkey Trace Start/Stop

Nsight Systems Workstation Edition can use hotkeys to control profiling. Press the hotkey to start and/or stop a trace session from within the target application’s graphic window. This is useful when tracing games and graphic applications that use fullscreen display. In these scenarios, switching to Nsight Systems’ UI would unnecessarily introduce the window manager’s footprint into the trace. To enable the use of Hotkey, check the Hotkey checkbox in the project settings page:

> 

The default hotkey is F12.

**Changing the Default Hotkey Binding** - A different hotkey binding can be configured by setting the `HotKeyIntValue` configuration field in the `config.ini` file.

Set the decimal numeric identifier of the hotkey you would like to use for triggering start/stop from the target app graphics window. The default value is 123 which corresponds to 0x7B, or the F12 key.

Virtual key identifiers are detailed in [MSDN’s Virtual-Key Codes](https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes).

Note that you must convert the hexadecimal values detailed in this page to their decimal counterpart before using them in the file. For example, to use the F1 key as a start/stop trace hotkey, use the following settings in the `config.ini` file:

``` cpp
HotKeyIntValue=112
```

#### Target Sampling Options on Windows

> 

Nsight Systems can sample one process tree. Sampling here means interrupting each processor periodically. The sampling rate is defined in the project settings and is either 100Hz, 1KHz (default value), 2Khz, 4KHz, or 8KHz.

> 

On Windows, Nsight Systems can collect thread activity of one process tree. Collecting thread activity means that each thread context switch event is logged and (optionally) a backtrace is collected at the point that the thread is scheduled back for execution. Thread states are displayed on the timeline.

If it was collected, the thread backtrace is displayed when hovering over a region where the thread execution is blocked.

#### Symbol Locations

Symbol resolution happens on host, and therefore does not affect performance of profiling on the target.

Press the **Symbol locations…** button to open the **Configure debug symbols location** dialog.

> 

Use this dialog to specify:

- Paths of PDB files

- Symbols servers

- The location of the local symbol cache

To use a symbol server:

1.  Install **Debugging Tools for Windows**, a part of the [Windows 10 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk).

2.  Add the symbol server URL using the **Add Server** button.

    Information about Microsoft’s public symbol server, which enables getting Windows operating system related debug symbols can be found [here](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/microsoft-public-symbols).

### Profiling QNX Targets from the GUI

Profiling on QNX devices is similar to the profiling on Linux devices. Please refer to the [Profiling Linux Targets from the GUI](index.html#profiling-linux-targets-from-the-gui) section for the detailed documentation. The major differences on the platforms are listed below:

- Backtrace sampling is not supported. Instead backtraces are collected for long OS runtime libraries calls. Please refer to the [OS Runtime Libraries Trace](index.html#os-runtime-libraries-trace) section for the detailed documentation.

- CUDA support is limited to CUDA 9.0+.

- Filesystem on QNX device might be mounted read-only. In that case Nsight Systems is not able to install target-side binaries, required to run the profiling session. Please make sure that target filesystem is writable before connecting to QNX target. For example, make sure the following command works:

  ``` cpp
  echo XX > /xx && ls -l /xx
  ```

### Nsight Streamer for Nsight Systems

A self-hosted NVIDIA Nsight Systems GUI running inside a Docker container enables remote access through a web browser. This configuration is particularly useful for analyzing data on remote servers or clusters.

For more information and instructions on running the container, visit: [Nsight Streamer for Nsight Systems on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys)

### GUI VNC container

Nsight Systems provides a build script to build a self-isolated Docker container with the Nsight Systems GUI and VNC server.

You can find the `build.py` script in the `host-linux-x64/Scripts/VncContainer` directory (or similar on other architectures) under your Nsight Systems installation directory. You will need to have [Docker](https://www.docker.com), and Python 3.5 or later.

**Available Parameters**

| Short Name | Full Name | Description |
|----|----|----|
|  | `--vnc-password` | (optional) Default password for VNC access (at least 6 characters). If it is specified and empty user will be asked during the build. Can be changed when running a container. |
| -aba | `--additional-build-arguments` | (optional) Additional arguments, which will be passed to the `docker`` ``build` command. |
| -hd | `--nsys-host-directory` | (optional) The directory with Nsight Systems host binaries (with GUI). |
| -td | `--nsys-target-directory` | (optional, repeatable) The directory with Nsight Systems target binaries (can be specified multiple times). |
|  | `--tigervnc` | (optional) Use TigerVNC instead of x11vnc. |
|  | `--http` | (optional) Install noVNC in the Docker container for HTTP access. |
|  | `--rdp` | (optional) Install xRDP in the Docker for RDP access. |
|  | `--geometry` | (optional) Default VNC server resolution in the format WidthxHeight (default 1920x1080). |
|  | `--build-directory` | (optional) The directory to save temporary files (with the write access for the current user). By default, script or tmp directory will be used. |

**Ports**

These ports can be published from the container to provide access to the Docker container:

| Port | Purpose | Condition |
|----|----|----|
| TCP 5900 | Port for VNC access |  |
| TCP 80 (optional) | Port for HTTP access to noVNC server | Container is build with `--http` parameter |
| TCP 3389 (optional) | Port for RDP access | Container is build with `--rdp` parameter |

**Volumes**

| Docker folder | Purpose | Description |
|----|----|----|
| /mnt/host | Root path for shared folders | Folder owned by the Docker user (inner content can be accessed from Nsight Systems GUI) |
| /mnt/host/Projects |  | Folder with projects and reports, created by Nsight Systems UI in container |
| /mnt/host/logs | Folder with inner services logs | May be useful to send reports to developers |

**Environment variables**

| Variable Name      | Purpose                                         |
|--------------------|-------------------------------------------------|
| VNC_PASSWORD       | Password for VNC access (at least 6 characters) |
| NSYS_WINDOW_WIDTH  | Width of VNC server display (in pixels)         |
| NSYS_WINDOW_HEIGHT | Height of VNC server display (in pixels)        |

**Examples**

With VNC access on port 5916:

``` cpp
sudo docker run -p 5916:5900/tcp -ti nsys-ui-vnc:1.0
```

With VNC access on port 5916 and HTTP access on port 8080:

``` cpp
sudo docker run -p 5916:5900/tcp -p 8080:80/tcp -ti nsys-ui-vnc:1.0
```

With VNC access on port 5916, HTTP access on port 8080, and RDP access on port 33890:

``` cpp
sudo docker run -p 5916:5900/tcp -p 8080:80/tcp -p 33890:3389/tcp -ti nsys-ui-vnc:1.0
```

With VNC access on port 5916, shared “HOME” folder from the host, VNC server resolution 3840x2160, and custom VNC password:

``` cpp
sudo docker run -p 5916:5900/tcp -v $HOME:/mnt/host/home -e NSYS_WINDOW_WIDTH=3840 -e NSYS_WINDOW_HEIGHT=2160 -e VNC_PASSWORD=7654321 -ti nsys-ui-vnc:1.0
```

With VNC access on port 5916, shared “HOME” folder from the host, and the projects folder to access reports created by the Nsight Systems GUI in container:

``` cpp
sudo docker run -p 5916:5900/tcp -v $HOME:/mnt/host/home -v /opt/NsysProjects:/mnt/host/Projects -ti nsys-ui-vnc:1.0
```

## Profiling within JupyterLab

The JupyterLab Nsight extension integrates Nsight Systems profiling into JupyterLab for profiling of Jupyter notebook cells. CUDA kernels launched by the cells as well as CUDA and Python code execution can be profiled and analyzed.

For more information and to install the extension, go to [JupyterLab Nsight extension on PyPI](https://pypi.org/project/jupyterlab-nvidia-nsight/)

**Basic usage of JupyterLab Nsight extension**

- Install the extension by running `pip`` ``install`` ``jupyterlab-nvidia-nsight`.  
  - Nsight Systems is not bundled with this extension. It should be installed separately.

- Launch (or restart) JupyterLab.

- Set Nsight Systems installation location in the extension settings (NVIDIA Nsight –\> Settings…).  
  - Leave this setting empty if Nsight Systems CLI executable is already in the system path.

- Open a notebook and enable Nsight Systems (NVIDIA Nsight –\> Profiling with Nsight Systems…).  
  - Set the desired options for nsys launch command (e.g., `--trace=cuda,nvtx,cublas,cudnn`).

  - This restarts the JupyterLab kernel.

  - A new green arrow icon appears in the notebook toolbar, and can be used to profile cells execution.

- (Optional) Open the generated report file in Nsight Systems GUI inside JupyterLab by double clicking on the report file.

**Fallback Example** - How to use Nsight Systems to profile code in individual cells of a Jupyter notebook when the extension is not available.

- Launch jupyter-lab with Nsight Systems using the desired trace options. For example:

  > nsys launch –trace=cuda,nvtx,cublas,cudnn jupyter lab

- (optional) Add NVTX ranges to the important operations in the notebook using range_push and range_pop. These NVTX ranges add information but are not used to define the profiling capture.

- To profile a cell, add a shell command to `nsys`` ``start` at the top of the cell and a shell command to `nsys`` ``stop` at the bottom of the cell. We recommend using the the absolute path to `nsys` on your system to make sure it is found.

- Save the notebook.

- Run all the cells required for the code you want to profile, then run the cell you want to profile.

- Each time the cell with `nsys`` ``start` and `nsys`` ``stop` is run, a new .nsys-rep file will be generated.

- Open the nsys-rep file in nsys-ui.

## Container, Scheduler, and Cloud Support

### Collecting Data Within a Container

While examples in this section use Docker container semantics, other containers work much the same.

The following information assumes the reader is knowledgeable regarding Docker containers. For further information about Docker use in general, see the [Docker documentation](https://docs.docker.com).

We strongly recommend using the CLI to profile in a container. Best container practice is to split services across containers when they do not require colocation. The Nsight Systems GUI is not needed to profile and brings in many dependencies, so the CLI is recommended. If you wish, the GUI can be in a separate side-car container you use to view your report. All you need is a shared folder between the containers. See section on [GUI VNC container](#gui-vnc-container) for more information.

#### Enable Docker Collection

When starting the Docker to perform a Nsight Systems collection, additional steps are required to enable the `perf_event_open` system call. This is required in order to utilize the Linux kernel’s perf subsystem which provides sampling information to Nsight Systems.

There are three ways to enable the `perf_event_open` syscall. You can enable it by using the `--privileged=true` switch, adding `--cap-add=SYS_ADMIN` switch to your docker run command file, or you can enable it by setting the seccomp security profile if your system meets the requirements.

Secure computing mode (seccomp) is a feature of the Linux kernel that can be used to restrict an application’s access. This feature is available only if the kernel is enabled with seccomp support. To check for seccomp support:

``` cpp
$ grep CONFIG_SECCOMP= /boot/config-$(uname -r)
```

The official Docker documentation says:

``` cpp
"Seccomp profiles require seccomp 2.2.1 which is not available on Ubuntu
14.04, Debian Wheezy, or Debian Jessie. To use seccomp on these distributions,
you must download the latest static Linux binaries (rather than packages)."
```

Download the default seccomp profile file, `default.json`, relevant to your Docker version. If `perf_event_open` is already listed in the file as guarded by `CAP_SYS_ADMIN`, then remove the `perf_event_open` line. Add the following lines under “syscalls” and save the resulting file as `default_with_perf.json`.

``` cpp
{
    "name": "perf_event_open",
    "action": "SCMP_ACT_ALLOW",
    "args": []
},
```

Then you will be able to use the following switch when starting the Docker to apply the new seccomp profile.

``` cpp
--security-opt seccomp=default_with_perf.json
```

#### Launch Docker Collection

Here is an example command that has been used to launch a Docker for testing with Nsight Systems:

``` cpp
sudo nvidia-docker run --network=host --security-opt seccomp=default_with_perf.json --rm -ti caffe-demo2 bash
```

There is a known issue where Docker collections terminate prematurely with older versions of the driver and the CUDA Toolkit. If collection is ending unexpectedly, please update to the latest versions.

After the Docker has been started, use the Nsight Systems CLI to launch a collection within the Docker. The resulting file can be imported into the Nsight Systems host like any other CLI result.

### Profiling Services in the Cloud

Nsight Cloud is a set of utilities designed to simplify the process of launching and controlling NVIDIA tools in cloud environments. For more information and download see: [NVIDIA Nsight Cloud](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/collections/nsight-cloud).

#### Profiling Services Launched via Kubernetes (Nsight Operator)

Nsight Systems now can provide profiling via sidecar injection without need to modify your containers or k8/helm specs.

> 

Once the sidecar is enabled, the data collected data can be filtered by namespace or pod using Kubernetes labels, or within a container process by using command-line regex.

This functionality is compatible with various cloud service provider’s in-house managed Kubernetes variants including AKS, EKS, GKE, and OKE.

Documentation and download for this sidecar is available at [NGC Nsight Operator](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/helm-charts/nsight-operator).

#### Streaming GUI (Nsight Streamer)

A self-hosted NVIDIA Nsight Systems GUI running inside a Docker container enables remote access through a web browser. This configuration is particularly useful for analyzing data on remote servers or clusters.

For more information and instructions on running the container, visit: [Nsight Streamer for Nsight Systems on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys).

### Profiling Slurm Jobs Running in Containers (Enroot/Pyxis or OCI)

When jobs are launched through Slurm and executed inside a container runtime — either Pyxis/Enroot (common on NVIDIA and HPC clusters) or a generic OCI runtime (Docker, containerd, Podman, CRI-O) - Nsight Systems can still be used to profile the workload, but the CLI (`nsys`) itself must be made available *inside* the container that Slurm launches, not just on the host.

The recommended pattern is **“inject, don’t rebuild”**: keep `nsys` installed once on the bare-metal compute node and bind-mount it into the container at launch time, then wrap the in-container command with `nsys`` ``profile` exactly as you would in a non-containerized MPI/Slurm job. This avoids baking Nsight Systems into every container image and makes it easy to upgrade the profiler independently of the workload.

There are three steps, regardless of the container runtime:

1.  Install Nsight Systems (examples below use `2026.2.1`) on the bare-metal Slurm compute nodes.

2.  Bind-mount that install into the container at the same absolute path via the container runtime’s configuration.

3.  Invoke `nsys`` ``profile` by absolute path (or, if you prefer, extend the container’s `PATH` without replacing it). A wrapper script is only needed when you want per-rank logic.

See also the “Handling Application Launchers (mpirun, deepspeed, etc)” and “Enable Docker Collection” sections — the same rules about rank-based wrapper scripts, `%q{...}` output templates, and `perf_event_open` apply here.

#### Install Nsight Systems on the Target Compute Nodes

Install the same Nsight Systems build on every compute node that the Slurm job may be scheduled on, and use an identical path on all nodes, e.g.:

``` text
/opt/nvidia/nsight-systems/2026.2.1
```

A shared filesystem (NFS, Lustre, GPFS) works as well, as long as the path is readable on every node and on the container’s mount namespace after the bind mount is applied.

Verify on one node:

``` console
$ /opt/nvidia/nsight-systems/2026.2.1/bin/nsys --version
```

and confirm the printed version matches what you expect to bind-mount into the container.

Note

**Keep CLI and UI versions aligned.** The Nsight Systems GUI / `nsys-ui` used to open a report should be the **same or newer** than the CLI that produced it. Reports from a newer CLI opened in an older GUI are the common failure mode. The `.qdrep` → `.nsys-rep` format change was in 2021.4, so any 2021.4-or-newer GUI can at least load the file; whether all timeline features render correctly still depends on the GUI being ≥ CLI version. Pin one version across all compute nodes so the bind mount resolves consistently.

##### Host kernel requirements

Profiling from inside a container still depends on the host kernel, so make sure the host satisfies the normal Nsight Systems requirements:

- `perf_event_paranoid` is permissive enough on the host (typically `<=`` ``2`). You can verify readiness from inside the container with `nsys`` ``status`` ``--environment`.

- The container has access to `perf_event_open` — see “Enable Docker Collection” above. For OCI runtimes this means `--cap-add=SYS_ADMIN`, `--privileged`, or a custom seccomp profile that allows `perf_event_open`. Enroot launches containers unprivileged and, by default, does not install Docker’s restrictive seccomp profile on top of the host policy, so `perf_event_open` is typically usable without extra flags — but verify with `nsys`` ``status`` ``--environment` inside the container before assuming it.

- CUDA/NVIDIA driver libraries are already made available in the container by the runtime (NVIDIA Container Toolkit for OCI, `--container-mounts` + the NVIDIA hook for Pyxis/Enroot). The Nsight Systems bind mount does not change that.

#### Make `nsys` Visible Inside the Container

##### Option A — Pyxis/Enroot via `srun`` ``--container-mounts`

Pyxis (the Slurm SPANK plugin for Enroot) exposes Enroot container launches as `srun` flags. The `--container-mounts` flag accepts `SRC:DST[:FLAGS][,SRC:DST...]` (multiple flags are joined with `+`, e.g. `ro+rprivate`). Bind-mount the host install into the container and invoke `nsys` by absolute path so nothing in the image environment is modified:

``` bash
srun \
  --container-image=nvcr.io#nvidia/pytorch:<tag> \
  --container-mounts=/opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
  /opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile \
    -t cuda,nvtx,mpi \
    -o /reports/run_%q{SLURM_PROCID}_%p \
    python train.py
```

Key points:

- `--container-mounts=<host>:<container>:ro` bind-mounts the host install read-only. `ro` is enough — the profiler does not write inside its own install tree.

- Calling `nsys` via its absolute in-container path avoids touching the image’s `PATH` at all. If you prefer `nsys` on `PATH` (for example for interactive use), append it *inside* the container rather than replacing the image `PATH`:

  ``` bash
  srun \
    --container-image=nvcr.io#nvidia/pytorch:<tag> \
    --container-mounts=/opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
    --export=ALL,NSYS_PATH=/opt/nvidia/nsight-systems/2026.2.1/bin \
    --container-env=NSYS_PATH \
    bash -lc 'PATH="$PATH:$NSYS_PATH"; nsys profile -o /reports/run_%q{SLURM_PROCID}_%p python train.py'
  ```

- If you need `nsys` reports to land on a shared filesystem, also bind-mount the output directory (e.g. `--container-mounts=...,/scratch/$USER/reports:/reports:rw`) and pass `-o`` ``/reports/...` to `nsys`` ``profile`.

Before adding `nsys` to the command line, do a quick container-runtime preflight:

``` bash
srun --container-image=nvcr.io#nvidia/pytorch:<tag> /bin/true
```

If this fails, fix the base Pyxis/Enroot launch path first, then add `--container-mounts` and profiling.

##### Option B — Enroot `mounts.d` + `environ.d` (recommended, image-independent)

When you do not control the `srun` invocation (for example a fleet-wide Slurm setup), Enroot can inject the mount and extend `PATH` via its standard configuration directories under `ENROOT_SYSCONF_PATH` (default `/etc/enroot`). This is the idiomatic Enroot approach and does not require a custom hook script.

`/etc/enroot/mounts.d/nsys.fstab`:

``` text
# Bind-mount the host |product-name| install into every Enroot container.
/opt/nvidia/nsight-systems/2026.2.1 /opt/nvidia/nsight-systems/2026.2.1 none bind,ro,nosuid,nodev 0 0
```

`/etc/enroot/environ.d/nsys.env`:

``` text
# Make `nsys` visible on PATH inside every Enroot container.
PATH=/opt/nvidia/nsight-systems/2026.2.1/bin:${PATH}
```

Both files are applied automatically to every container launched via Enroot (and therefore via Pyxis). No image changes are needed.

##### Option C — Enroot pre-start hook (only if you need logic)

Use a hook script only when you need conditional behavior (for example, pick a different Nsight Systems install based on the image or the host’s GPU family). Hooks live in `/etc/enroot/hooks.d/` with a `.sh` extension and run with full capabilities before the container switches to its final root. They receive `ENROOT_ROOTFS`, `ENROOT_MOUNTS`, and `ENROOT_ENVIRON`.

`/etc/enroot/hooks.d/50-nsys.sh`:

``` bash
#!/usr/bin/env bash
# Append a bind mount for |product-name| to the container's mount file,
# and add it to PATH via the environ file. Prefer mounts.d / environ.d
# unless you actually need logic here.

set -euo pipefail

: "${ENROOT_MOUNTS:?not set}"
: "${ENROOT_ENVIRON:?not set}"

NSYS_HOST_PREFIX="/opt/nvidia/nsight-systems/2026.2.1"
NSYS_IN_CONTAINER="/opt/nvidia/nsight-systems/2026.2.1"

echo "${NSYS_HOST_PREFIX} ${NSYS_IN_CONTAINER} none bind,ro,nosuid,nodev 0 0" \
    >> "${ENROOT_MOUNTS}"
echo "PATH=${NSYS_IN_CONTAINER}/bin:\${PATH}" >> "${ENROOT_ENVIRON}"
```

##### Option D — OCI runtime (Docker / Podman / containerd / CRI-O)

For containers launched through an OCI runtime under Slurm (for example via `srun`` ``--container-...` wired to a Docker or container orchestrator, or via a site-specific launcher), declare the bind mount in the container configuration and invoke `nsys` by its absolute in-container path so the image’s own `PATH` and other environment variables are preserved.

Docker / Podman flags on the launch command:

``` bash
docker run --rm --gpus all \
  --cap-add=SYS_ADMIN \
  -v /opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
  my-workload:latest \
  /opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile \
    -t cuda,nvtx,mpi \
    -o /reports/run_%p \
    python train.py
```

Or in the OCI runtime `config.json` (for a container spec consumed by `runc`/`crun`):

``` json
{
  "process": {
    "args": [
      "/opt/nvidia/nsight-systems/2026.2.1/bin/nsys",
      "profile",
      "-t", "cuda,nvtx,mpi",
      "-o", "/reports/run_%p",
      "python", "train.py"
    ]
  },
  "mounts": [
    {
      "destination": "/opt/nvidia/nsight-systems/2026.2.1",
      "type": "bind",
      "source": "/opt/nvidia/nsight-systems/2026.2.1",
      "options": ["rbind", "ro"]
    }
  ]
}
```

Notes for OCI runtimes:

- Add `--cap-add=SYS_ADMIN` (or the custom seccomp profile described in “Enable Docker Collection”) so `perf_event_open` is not blocked.

- Calling `nsys` by absolute path means the image’s own `PATH` (and any other `ENV` set by the image) is preserved. If you’d rather have `nsys` on `PATH`, append inside the container (`PATH="$PATH:/opt/nvidia/nsight-systems/2026.2.1/bin"`) instead of replacing `PATH` via `-e` / `process.env`.

- GPU exposure is still the runtime’s responsibility (`--gpus`` ``all` / NVIDIA Container Toolkit hook / `nvidia.com/gpu` device plugin). The Nsight Systems mount does not change that.

- If `-o` writes to a host directory, add a second bind mount for it (`-v`` ``/scratch/$USER/reports:/reports:rw` or a corresponding `mounts` entry) — the profiler cannot write through a read-only mount.

#### Wrap the In-Container Command with `nsys`` ``profile`

Once the Nsight Systems install is bind-mounted, the invocation is identical to the non-container Slurm + MPI pattern. In multi-node jobs, the launcher runs outside the profiler and `nsys`` ``profile` wraps the per-rank program:

``` bash
srun [srun args] \
  --container-image=... --container-mounts=... \
  /opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile -t cuda,nvtx,mpi \
    -o /reports/run_%q{SLURM_PROCID}_%p \
    ./myapp [app args]
```

Use `%q{SLURM_PROCID}` (Slurm) / `%q{OMPI_COMM_WORLD_RANK}` (Open MPI) / `%q{PMI_RANK}` (MPICH) or `%p` in `-o` so concurrent ranks do not clobber each other’s report file. `%h` (hostname) and `%%` (literal `%`) are also supported.

Warning

An error will occur if several processes want to write to the same report file at the same time.

##### Profiling only a subset of ranks

To reduce trace volume, profile just a representative rank. Put a wrapper script inside the container (or bind-mounted in) and call it instead of the binary. Example `nsys_profile.sh` for Slurm:

``` bash
#!/bin/bash
# Profile only the node-local rank 0. Use $SLURM_PROCID for global rank 0.
NSYS_PATH="${NSYS_PATH:-/opt/nvidia/nsight-systems/2026.2.1/bin}"
PATH="${PATH}:${NSYS_PATH}"
if [ "${SLURM_LOCALID}" -eq 0 ]; then
  nsys profile -t cuda,nvtx,mpi \
    -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 \
    -o /reports/run_%q{SLURM_PROCID}_%p \
    "$@"
else
  "$@"
fi
```

Launch:

``` bash
srun --container-image=... \
     --container-mounts=/opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro,/scratch/$USER/reports:/reports:rw \
     --export=ALL,NSYS_PATH=/opt/nvidia/nsight-systems/2026.2.1/bin \
     --container-env=NSYS_PATH \
     ./nsys_profile.sh python train.py
```

Note

If only a subset of MPI ranks is profiled, set `NSYS_MPI_STORE_TEAMS_PER_RANK=1` so all members of custom MPI communicators are stored per rank. Otherwise the run may hang or fail with an MPI error. If communicator tracking itself is the problem, it can be disabled with `NSYS_MPI_DISABLE_COMMUNICATOR_TRACKING=1`.

##### GPU and NIC metrics with multiple ranks per node

As in the non-container case, if more than one rank per node would enable GPU/NIC metric collection, restrict it to one rank to avoid contention:

``` bash
#!/bin/bash
if [ "${SLURM_LOCALID}" -eq 0 ]; then
  nsys profile --nic-metrics=lf --gpu-metrics-devices=all "$@"
else
  nsys profile "$@"
fi
```

Or one rank per GPU, with metrics scoped to that GPU:

``` bash
#!/bin/bash
nsys profile -e CUDA_VISIBLE_DEVICES=${SLURM_LOCALID} \
             --gpu-metrics-devices=${SLURM_LOCALID} "$@"
```

#### Troubleshooting checklist

| Symptom | Likely cause | Fix |
|----|----|----|
| `nsys:`` ``command`` ``not`` ``found` inside container | Bind mount missing, or `nsys` not on `PATH` | Verify `ls`` ``/opt/nvidia/nsight-systems/<ver>/bin/nsys` inside the container. Prefer calling `nsys` by absolute path; if you need it on `PATH`, append inside the container (`PATH="$PATH:.../bin"`). |
| `nsys`` ``--version` reports an unexpected release | Wrong host install picked up, or a floating symlink | Mount an explicit versioned prefix (`/opt/nvidia/nsight-systems/2026.2.1`), not `/opt/nvidia/nsight-systems/current`. |
| `nsys`` ``status`` ``--environment` reports sampling disabled, or `Failed`` ``to`` ``create`` ``perf`` ``event` | `perf_event_open` blocked by container or host | OCI: add `--cap-add=SYS_ADMIN` or the seccomp profile from “Enable Docker Collection”. Enroot: ensure host `perf_event_paranoid`` ``<=`` ``2` and no site seccomp is shadowing it. |
| Reports not visible after the job ends | Output path is container-only | Point `-o` at a writable bind mount (e.g. `/reports` backed by `/scratch/$USER/reports`). |
| Several ranks overwrite the same `.nsys-rep` | `-o` not rank-templated | Use `%q{SLURM_PROCID}` / `%q{OMPI_COMM_WORLD_RANK}` / `%q{PMI_RANK}` / `%p` in `-o`. |
| Profiler version mismatch across nodes | Different host installs per node | Pin the same version at the same path on every node, or use a shared filesystem. |
| `pyxis:`` ``failed`` ``to`` ``import`` ``docker`` ``image` with `mkdir:`` ``cannot`` ``create`` ``directory`` ``'/run/enroot':`` ``Permission`` ``denied` | Enroot runtime dir not provisioned or not writable for the job user | Ensure `/run/enroot` exists and is world-traversable with a per-user `/run/enroot/user-<uid>`, or set `ENROOT_RUNTIME_PATH` to a writable path. Quick dev setup: `sudo`` ``install`` ``-d`` ``-m`` ``1777`` ``/run/enroot`` ``&&`` ``sudo`` ``install`` ``-d`` ``-m`` ``700`` ``-o`` ``"$USER"`` ``-g`` ``"$USER"`` ``/run/enroot/user-$(id`` ``-u)`. |
| `enroot`/`pyxis` import fails with `Could`` ``not`` ``process`` ``JSON`` ``input` | Registry endpoint resolved to a web page (e.g. `docker.io` redirects to `www.docker.com`) | Use an explicit registry: `docker://registry-1.docker.io/library/ubuntu:22.04` or Pyxis form `registry-1.docker.io#library/ubuntu:22.04`. |
| GUI refuses to open a report produced by the container | GUI older than the CLI that recorded the report | Upgrade the Nsight Systems GUI to match or exceed the CLI version. |

## Custom ETW Trace

Use the custom ETW trace feature to enable and collect any manifest-based ETW log. The collected events are displayed on the timeline on dedicated rows for each event type.

Custom ETW is available on Windows target machines.

> 

> 



To retain the .etl trace files captured, so that they can be viewed in other tools (e.g., GPUView), change the **Save ETW log files in project folder** option under **Profile Behavior** in Nsight Systems’s global Options dialog. The .etl files will appear in the same folder as the .nsys-rep file, accessible by right-clicking the report in the Project Explorer and choosing **Show in Folder…**. Data collected from each ETW provider will appear in its own .etl file, and an additional .etl file named `Report`` ``XX-Merged-\*.etl`, containing the events from all captured sources, will be created as well.

## Direct3D Trace

Nsight Systems has the ability to trace both the Direct3D 11 API and the Direct3D 12 API on Windows targets.

### D3D11 API trace

Nsight Systems can capture information about Direct3D 11 API calls made by the profiled process. This includes capturing the execution time of D3D11 API functions, performance markers, and frame durations.

> 

### D3D12 API Trace

Direct3D 12 is a low-overhead 3D graphics and compute API for Microsoft Windows. Information about Direct3D 12 can be found at the [Direct3D 12 Programming Guide](https://docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide).

Nsight Systems can capture information about Direct3D 12 usage by the profiled process. This includes capturing the execution time of D3D12 API functions, corresponding workloads executed on the GPU, performance markers, and frame durations.



The Command List Creation row displays time periods when command lists were being created. This enables developers to improve their application’s multi-threaded command list creation. Command list creation time period is measured between the call to `ID3D12GraphicsCommandList::Reset` and the call to `ID3D12GraphicsCommandList::Close`.



The GPU row shows a compressed view of the D3D12 queue activity, color-coded by the queue type. Expanding it will show the individual queues and their corresponding API calls.



A Command Queue row is displayed for each D3D12 command queue created by the profiled application. The row’s header displays the queue’s running index and its type (Direct, Compute, Copy).



The DX12 API Memory Ops row displays all API memory operations and non-persistent resource mappings. Event ranges in the row are color-coded by the heap type they belong to (Default, Readback, Upload, Custom, or CPU-Visible VRAM), with usage warnings highlighted in yellow. A breakdown of the operations can be found by expanding the row to show rows for each individual heap type.

The following operations and warnings are shown:

- Calls to `ID3D12Device::CreateCommittedResource`, `ID3D12Device4::CreateCommittedResource1`, and `ID3D12Device8::CreateCommittedResource2`

  - A warning will be reported if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the method’s `HeapFlags` parameter.

- Calls to `ID3D12Device::CreateHeap` and `ID3D12Device4::CreateHeap1`

  - A warning will be reported if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the `Flags` field of the method’s `pDesc` parameter

- Calls to `ID3D12Resource::ReadFromSubResource`

  - A warning will be reported if the read is to a `D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE` CPU page or from a `D3D12_HEAP_TYPE_UPLOAD` resource.

- Calls to `ID3D12Resource::WriteToSubResource`

  - A warning will be reported if the write is from a `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK` CPU page or to a `D3D12_HEAP_TYPE_READBACK` resource.

- Calls to `ID3D12Resource::Map` and `ID3D12Resource::Unmap` will be matched into \[Map, Unmap\] ranges for non-persistent mappings. If a mapping range is nested, only the most external range (reference count = 1) will be shown.



The API row displays time periods where `ID3D12CommandQueue::ExecuteCommandLists` was called. The GPU Workload row displays time periods where workloads were executed by the GPU. The workload’s type (Graphics, Compute, Copy, etc.) is displayed on the bar representing the workload’s GPU execution.



In addition, you can see the PIX command queue CPU-side performance markers, GPU-side performance markers, and the GPU Command List performance markers, each in their row.

  

Clicking on a GPU workload highlights the corresponding `ID3D12CommandQueue::ExecuteCommandLists`, `ID3D12GraphicsCommandList::Reset` and `ID3D12GraphicsCommandList::Close`` ``API` calls, and vice versa.



Detecting which CPU thread was blocked by a fence can be difficult in complex apps that run tens of CPU threads. The timeline view displays the 3 operations involved:

- The CPU thread pushing a signal command and fence value into the command queue. This is displayed on the DX12 Synchronization sub-row of the calling thread.

- The GPU executing that command, setting the fence value and signaling the fence. This is displayed on the GPU Queue Synchronization sub-row.

- The CPU thread calling a Win32 wait API to block-wait until the fence is signaled. This is displayed on the Thread’s OS runtime libraries row.

Clicking one of these will highlight it and the corresponding other two calls.



Nsight Systems D3D12 trace captures D3D12 Work Graphs dispatch calls to DispatchGraph and time boundaries of the GPU execution of the work graph.



The DX12 API row displays `ID3D12GraphicsCommandList10::DispatchGraph` calls. The GPU PIX Markers row marks graph execution by the GPU with a custom marker captioned “D3D12 graph execution.”



## WDDM Queues

The Windows Display Driver Model (WDDM) architecture uses queues to send work packets from the CPU to the GPU. Each D3D device in each process is associated with one or more contexts. Graphics, compute, and copy commands that the profiled application uses are associated with a context, batched in a command buffer, and pushed into the relevant queue associated with that context.

Nsight Systems can capture the state of these queues during the trace session.

Enabling the “Extensive trace” option will also capture extended DxgKrnl events from the `Microsoft-Windows-DxgKrnl` provider, such as Hardware Scheduling queues, context status, allocations, sync wait, signal events, etc.

> 

A command buffer in a WDDM queues may have one the following types:

- Render

- Deferred

- System

- MMIOFlip

- Wait

- Signal

- Device

- Software

It may also be marked as a Present buffer, indicating that the application has finished rendering and requests to display the source surface.

See the Microsoft documentation for the WDDM architecture and the `DXGKETW_QUEUE_PACKET_TYPE` enumeration.

To retain the .etl trace files captured, so that they can be viewed in other tools (e.g. GPUView), change the “Save ETW log files in project folder” option under “Profile Behavior” in Nsight Systems’s global Options dialog. The .etl files will appear in the same folder as the .nsys-rep file, accessible by right-clicking the report in the Project Explorer and choosing “Show in Folder…”. Data collected from each ETW provider will appear in its own .etl file, and an additional .etl file named “Report XX-Merged-\*.etl”, containing the events from all captured sources, will be created as well.

## WDDM HW Scheduler

When GPU Hardware Scheduling is enabled in Windows 10 or newer, the Windows Display Driver Model (WDDM) uses the `DxgKrnl` ETW provider to expose report of NVIDIA GPUs’ hardware scheduling context switches.

Nsight Systems can capture these context switch events, and display under the GPUs in the timeline rows titled WDDM HW Scheduler - \[HW Queue type\]. The ranges under each queue will show the process name and PID assoicated with the GPU work during the time period.

The events will be captured if GPU Hardware Scheduling is enabled in the Windows System Display settings, and “Collect WDDM Trace” is enabled in the Nsight Systems Project Settings.

> 

## Vulkan API Trace

### Vulkan Overview

Vulkan is a low-overhead, cross-platform 3D graphics and compute API, targeting a wide variety of devices from PCs to mobile phones and embedded platforms. The Vulkan API is defined by the Khronos Group. Information about Vulkan and the Khronos Group can be found at the [Khronos Vulkan Site](https://www.khronos.org/vulkan/).

Nsight Systems can capture information about Vulkan usage by the profiled process. This includes capturing the execution time of Vulkan API functions, corresponding GPU workloads, debug util labels, and frame durations. Vulkan profiling is supported on both Windows and x86 Linux operating systems.



The Command Buffer Creation row displays time periods when command buffers were being created. This enables developers to improve their application’s multi-threaded command buffer creation. Command buffer creation time period is measured between the call to `vkBeginCommandBuffer` and the call to `vkEndCommandBuffer`.



A Queue row is displayed for each Vulkan queue created by the profiled application. The API sub-row displays time periods where `vkQueueSubmit` was called. The GPU Workload sub-row displays time periods where workloads were executed by the GPU.



In addition, you can see [Vulkan debug util labels](https://github.com/KhronosGroup/Vulkan-Docs/blob/master/appendices/VK_EXT_debug_utils.txt) on both the CPU and the GPU.



Clicking on a GPU workload highlights the corresponding `vkQueueSubmit` call, and vice versa.



The Vulkan Memory Operations row contains an aggregation of all the Vulkan host-side memory operations, such as host-blocking writes and reads or non-persistent map-unmap ranges.

The row is separated into sub-rows by heap index and memory type - the tooltip for each row and the ranges inside show the heap flags and the memory property flags.

 

### Pipeline Creation Feedback

When tracing target application calls to Vulkan pipeline creation APIs, Nsight Systems leverages the Pipeline Creation Feedback extension to collect more details about the duration of individual pipeline creation stages.

See [Pipeline Creation Feedback extension](https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VkPipelineCreationFeedbackEXT) for details about this extension.

Vulkan pipeline creation feedback is available on NVIDIA driver release 435 or later.



### Vulkan GPU Trace Notes

- Vulkan GPU trace is available only when tracing apps that use NVIDIA GPUs.

- The endings of Vulkan Command Buffers execution ranges on Compute and Transfer queues may appear earlier on the timeline than their actual occurrence.

## Stutter Analysis

Stutter Analysis Overview

Nsight Systems on Windows targets displays stutter analysis visualization aids for profiled graphics applications that use either OpenGL, D3D11, D3D12 or Vulkan, as detailed below in the following sections.

### FPS Overview

The Frame Duration section displays frame durations on both the CPU and the GPU.

> 

The frame duration row displays live FPS statistics for the current timeline viewport. Values shown are:

1.  Number of CPU frames shown of the total number captured.

2.  Average, minimal, and maximal CPU frame time of the currently displayed time range.

3.  Average FPS value for the currently displayed frames.

4.  The 99th percentile value of the frame lengths (such that only 1% of the frames in the range are longer than this value).

The values will update automatically when scrolling, zooming or filtering the timeline view.

> 

The stutter row highlights frames that are significantly longer than the other frames in their immediate vicinity.

The stutter row uses an algorithm that compares the duration of each frame to the median duration of the surrounding 19 frames. Duration difference under 4 milliseconds is never considered a stutter, to avoid cluttering the display with frames whose absolute stutter is small and not noticeable to the user.

For example, if the stutter threshold is set at 20%:

1.  Median duration is 10 ms. Frame with 13 ms time will not be reported (relative difference \> 20%, absolute difference \< 4 ms).

2.  Median duration is 60 ms. Frame with 71 ms time will not be reported (relative difference \< 20%, absolute difference \> 4 ms).

3.  Median duration is 60 ms. Frame with 80 ms is a stutter (relative difference \> 20%, absolute difference \> 4 ms, both conditions met).

**OSC detection**

The “19 frame window median” algorithm by itself may not work well with some cases of “oscillation” (consecutive fast and slow frames), resulting in some false positives. The median duration is not meaningful in cases of oscillation and can be misleading.

To address the issue and identify if oscillating frames, the following method is applied:

1.  For every frame, calculate the median duration, 1st and 3rd quartiles of 19-frames window.

2.  Calculate the delta and ratio between 1st and 3rd quartiles.

3.  If the 90th percentile of 3rd - 1st quartile delta array \> 4 ms AND the 90th percentile of 3rd/1st quartile array \> 1.2 (120%) then mark the results with “OSC” text.

Right-clicking the Frame Duration row caption lets you choose the target frame rate (30, 60, 90 or custom frames per second).



By clicking the **Customize FPS Display** option, a customization dialog pops up. In the dialog, you can now define the frame duration threshold to customize the view of the potentially problematic frames. In addition, you can define the threshold for the stutter analysis frames.



Frame duration bars are color-coded:

- Green, the frame duration is shorter than required by the target FPS ratio.

- Yellow, duration is slightly longer than required by the target FPS rate.

- Red, duration far exceeds that required to maintain the target FPS rate.



The CPU Frame Duration row displays the CPU frame duration measured between the ends of consecutive frame boundary calls:

- The OpenGL frame boundaries are `eglSwapBuffers/glXSwapBuffers/SwapBuffers` calls.

- The D3D11 and D3D12 frame boundaries are `IDXGISwapChainX::Present` calls.

- The Vulkan frame boundaries are `vkQueuePresentKHR` calls.

The timing of the actual calls to the frame boundary calls can be seen in the blue bar at the bottom of the CPU frame duration row

The GPU Frame Duration row displays the time measured between:

- The start time of the first GPU workload execution of this frame.

- The start time of the first GPU workload execution of the next frame.

**Reflex SDK**

NVIDIA Reflex SDK is a series of NVAPI calls that allow applications to integrate the Ultra Low Latency driver feature more directly into their game to further optimize synchronization between simulation and rendering stages and lower the latency between user input and final image rendering. For more details about Reflex SDK, see the [Reflex SDK Site](https://developer.nvidia.com/reflex/).

Nsight Systems will automatically capture NVAPI functions when either Direct3D 11, Direct3D 12, or Vulkan API trace are enabled.

The Reflex SDK row displays timeline ranges for the following types of latency markers:

- RenderSubmit

- Simulation

- Present

- Driver

- OS Render Queue

- GPU Render



**Performance Warnings row**

This row shows performance warnings and common pitfalls that are automatically detected based on the enabled capture types. Warnings are reported for:

- ETW performance warnings.

- Vulkan calls to `vkQueueSubmit` and D3D12 calls to `ID3D12CommandQueue::ExecuteCommandList` that take a longer time to execute than the total time of the GPU workloads they generated.

- [D3D12 Memory Operation warnings](index.html#d3d12-api-trace).

- Usage of Vulkan API functions that may adversely affect performance.

- Creation of a Vulkan device with memory zeroing, whether by physical device default or manually.

- Vulkan command buffer barrier which can be combined or removed, such as subsequent barriers or read-to-read barriers.



### Frame Health

The Frame Health row displays actions that took significantly a longer time during the current frame, compared to the median time of the same actions executed during the surrounding 19-frames. This is a great tool for detecting the reason for frame time stuttering. Such actions may be: shader compilation, present, memory mapping, and more. Nsight Systems measures the accumulated time of such actions in each frame. For example: calculating the accumulated time of shader compilations in each frame and comparing it to the accumulated time of shader compilations in the surrounding 19 frames.

Example of a Vulkan frame health row:

 

### Windows GPU Memory Utilization

Each GPU has two rows detailing its memory utilization: **GPU VRAM**, showing the memory consumed on the device, and **GPU WDDM SYSMEM**, showing the memory consumed on the host computer RAM.



These rows show a green-colored line graph for the memory budget for this memory segment, and an orange-colored line graph for the actual amount of memory used. Note that these graphs are scaled to fit the highest value enconutered, as indicated by the “Y axis” value in the row header. You can use the vertical zoom slider in the top-right of the timeline view to make the row taller and view the graph in more detail.



Note that the value in the GPU VRAM row is not the same as the CUDA kernel memory allocation graph, see [CUDA GPU Memory Allocation Graph](#cuda-gpu-memory-allocation-graph) for that functionality.

The GPU VRAM row also has several child rows, accessed by expanding the row in the tree view

The events will be captured if “Collect WDDM Trace” is enabled along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in the Nsight Systems Project Settings.



**VidMm Device Suspension**

This row displays time ranges when the GPU memory manager suspended all memory transfer operations, pending the completion of a single memory transfer.

The events will be captured if “Collect WDDM Trace” and “Extensive trace including Hardware Scheduling queues, context status, allocations, sync wait and signal events, etc.” are enabled in the Nsight Systems Project Settings.

**Demoted Memory**

This row displays the amount of VRAM that was demoted from GPU local memory to non-local memory (possibly due to exceeding the VRAM budget) as a blue-colored line graph. High amounts of demoted memory could be indicative of video memory leaks or poor memory management. Note that the Demoted memory row is scaled to its highest value, similar to the GPU VRAM and GPU WDDM SYSMEM rows.

The events will be captured if “Collect WDDM Trace” is enabled along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in the Nsight Systems Project Settings.

**Resource Allocations**



This row shows markers indicating resource allocation events. VRAM resources are shown as green markers while SYSMEM resources are shown in gray. Hovering over a marker or selecting it in the [Events view](index.html#events-view) will display all the allocation parameters as well as the call stack that led to the allocation event.

The events will be captured if “Collect WDDM Trace” is enabled along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in the Nsight Systems Project Settings.

**Resource Migrations**



This row displays a breakdown of resources’ movement between VRAM and SYSMEM, focusing on resource evictions. The main row shows a timeline of total evicted resource memory and count as a red-colored line graph.

Each child row displays a timeline of the status of each resource, as reflected by WDDM events related to it. If the object has been named using PIX or `ID3D11Object::SetName` / `ID3D12Object::SetName`, the name will be shown in the row title. Whether named or not, the row title will also show the resource dimensions, format, priority, and the memory size migrated. If the resource was migrated in parts using subresources, the row can be expanded to show the status for each subresource at any given time.

Expanding the row for a resource will show the individual WDDM events relevant to it and the call stacks that led to each event.

By default, the resources are sorted by Relevance (most / largest migrations). Right-clicking the main Resource Migrations row header allows choosing between the following sorting options:

- Relevance

- Name

- Format

- Priority

- Earliest allocation timestamp (order of appearance on the host)

- Earliest migration timestamp (order of appearance on the device)

The top 5 resources are shown initially. If more than 5 resources exist, a row showing the number of hidden resources and buttons allowing to show more or fewer of them will appear below them. Right-click this row and select “show all” or “show all collapsed” to display all the resources at once.

The events will be captured if “Collect WDDM Trace” is enabled along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in the Nsight Systems Project Settings. Additionally, to correlate Graphics API debug name events with resource migration events, the “Collect DX12” or “Collect Vulkan” option should be enabled.

**Memory Transfer**



This row shows an overview of all memory transfer operations. Device-to-host transfers are shown in orange, host-to-device transfers are shown in green, discarded device memory is shown in light green, and unknown events are shown in dark gray. The height of each event marker corresponds to the amount of memory that the event affected. Hovering over the marker will show the exact amount.

Expanding the row will show a breakdown of the events by each specific type.

The events will be captured if “Collect WDDM Trace” is enabled along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in the Nsight Systems Project Settings.

**System Committed VRAM**



This row represents the total size of committed VRAM by all processes currently using the GPU. The stacked chart displays colored layers. Each layer corresponds to the VRAM commitment of a specific process.

To track VRAM commitment, enable “Collect WDDM Trace” along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in Nsight Systems Project Settings.

**VRAM Resource Types Distribution**



This row shows the distribution of VRAM usage across different resource types per process. it is color-coded to show the different resource types, and the height of each segment corresponds to the amount of VRAM used by that resource type. Expand the chart’s parent row to expose detailed separate rows for individual resource categories.

The events will be captured if “Collect WDDM Trace” is enabled along with either “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” in the Nsight Systems Project Settings. Additionally, to correlate Graphics API debug name events with resource migration events, the “Collect DX12” or “Collect Vulkan” option should be enabled.

### Vertical Synchronization

The VSYNC rows display when the monitor’s vertical synchronizations occur.



## CUDA Trace

### Basic CUDA trace

Nsight Systems is capable of capturing information about CUDA execution in the profiled process.

The following information can be collected and presented on the timeline in the report:

- CUDA API trace — trace of CUDA Runtime and CUDA Driver calls made by the application.

  - CUDA Runtime calls typically start with `cuda` prefix (e.g. `cudaLaunch`).

  - CUDA Driver calls typically start with `cu` prefix (e.g. `cuDeviceGetCount`).

- CUDA workload trace — trace of activity happening on the GPU, which includes memory operations (e.g., Host-to-Device memory copies) and kernel executions. Within the threads that use the CUDA API, additional child rows will appear in the timeline tree.

- On Nsight Systems Workstation Edition, cuDNN and cuBLAS API tracing and OpenACC tracing.



Near the bottom of the timeline row tree, the GPU node will appear and contain a CUDA node. Within the CUDA node, each CUDA context used within the process will be shown along with its corresponding CUDA streams. Steams will contain memory operations and kernel launches on the GPU. Kernel launches are represented by blue, while memory transfers are displayed in red.



The easiest way to capture CUDA information is to launch the process from Nsight Systems, and it will setup the environment for you. To do so, simply set up a normal launch and select the **Collect CUDA trace** checkbox.

For Nsight Systems Workstation Edition this looks like:

> 

For Nsight Systems Embedded Platforms Edition this looks like:

> 

Additional configuration parameters are available:

- **System-wide CUDA trace** - collect CUDA trace from all processes on the system. By default, CUDA trace is captured from the target process and its descendants only. Select this option to switch to system-wide trace mode. Note: Only CUDA processes launched after the collection start and by the same user will be included. Only one session can collect system-wide CUDA trace at a time on the entire system; other sessions will fall back to process-tree scope.

- **Collect hardware-based trace** - switches Nsight Systems from an instrumented CUDA collection to a hardware based collection. For workloads that launch many short duration kernels, the overhead of kernel timestamp collection can be significantly reduced. **Note:** Works on Blackwell based or later GPUs. Does not support CUDA tracing of MPS workloads. Does not work in MIG partitions or virtual GPU environments (vGPU).

- **Collect backtraces for API calls longer than X seconds** - turns on collection of CUDA API backtraces and sets the minimum time a CUDA API event must take before its backtraces are collected. Setting this value too low can cause high application overhead and seriously increase the size of your results file.

- **Flush data periodically** — specifies the period after which an attempt to flush CUDA trace data will be made. Normally, in order to collect full CUDA trace, the application needs to finalize the device used for CUDA work (call `cudaDeviceReset()`, and then let the application gracefully exit (as opposed to crashing).

  This option allows flushing CUDA trace data even before the device is finalized. However, it might introduce additional overhead to a random CUDA Driver or CUDA Runtime API call.

- **Skip some API calls** — avoids tracing insignificant CUDA Runtime API calls (namely, `cudaConfigureCall()`, `cudaSetupArgument()`, `cudaHostGetDevicePointers()`). Not tracing these functions allows Nsight Systems to significantly reduce the profiling overhead, without losing any interesting data.

- **Collect GPU Memory Usage** - collects information used to generate a graph of CUDA allocated memory across time. Note that this will increase overhead. See [CUDA GPU Memory Allocation Graph](#cuda-gpu-memory-allocation-graph).

- **Collect Unified Memory CPU page faults** - collects information on page faults that occur when CPU code tries to access a memory page that resides on the device. See [Unified Memory CPU Page Faults](#unified-memory-cpu-page-faults).

- **Collect Unified Memory GPU page faults** - collects information on page faults that occur when GPU code tries to access a memory page that resides on the CPU. See [Unified Memory GPU Page Faults](#unified-memory-gpu-page-faults).

- **Collect CUDA Graph trace** - by default, CUDA tracing will collect and expose information on a whole graph basis. The user can opt to collect on a node per node basis. See [CUDA Graph Trace](#cuda-graph-trace).

- **Collect CUDA Event trace** - track device-side CUDA Event (the synchronization mechanism) completion, and get better correlation support among CUDA Event APIs. [CUDA Event Trace](#cuda-event-trace).

- For Nsight Systems Workstation Edition, **Collect cuDNN trace**, **Collect cuBLAS trace**, **Collect OpenACC trace** - selects which (if any) extra libraries that depend on CUDA to trace.

  OpenACC versions 2.0, 2.5, and 2.6 are supported when using PGI runtime version 15.7 or greater and not compiling statically. In order to differentiate constructs, a PGI runtime of 16.1 or later is required. Note that Nsight Systems Workstation Edition does not support the GCC implementation of OpenACC at this time.

Note

If your application crashes before all collected CUDA trace data has been copied out, some or all data might be lost and not present in the report.

Note

Nsight Systems will not have information about CUDA events that were still in device buffers when analysis terminated. It is a good idea, if using cudaProfilerAPI to control analysis to call cudaDeviceReset before ending analysis.

### Launching NVIDIA Nsight Compute from a CUDA Kernel

After you have used CUDA trace in Nsight Systems to locate a potential problem kernel, you may want to run NVIDIA Nsight Compute on that specific kernel. Right click on the kernel to bring up a menu.



If this is the first time that the user has selected this feature, then we show the following dialog box to get their preferences:



The first option invokes the NVIDIA Nsight Compute UI with known parameters. It is the preferred option for local or remote profiling. The user must provide the location of the ncu-ui executable. Nsight Systems will verify that the path and executable are valid.

The second option is provided for the convenience of users who do not have NVIDIA Nsight Compute installed on the host system, but simply want the command line they can use to run the Nsight Compute on the remote target by themselves without much automation.



If the user selects the option to start the NCU UI, Nsight Systems invokes it with any relevant parameters from the Nsight Systems run.

The screenshot below shows NCU UI invoked by Nsight Systems. The red circles indicate the parameters pre-populated by Nsight Systems. Users may modify any of these parameters before launching the application and profiling the selected kernel.



### CUDA GPU Memory Allocation Graph

When the **Collect GPU Memory Usage** option is selected from the **Collect CUDA trace** option set, Nsight Systems will track CUDA GPU memory allocations and deallocations and present a graph of this information in the timeline. This is not the same as the GPU memory graph generated during stutter analysis on the Windows target. See [Windows GPU Memory Utilization](#windows-gpu-memory-utilization).

Below, in the report on the left, memory is allocated and freed during the collection. In the report on the right, memory is allocated, but not freed during the collection.



Here is another example, where allocations are happening on multiple GPUs.



Nsight Systems uses CUPTI for CUDA profiling, including to collect the CUDA memory usage by the application processes. CUPTI tracks various kinds of memory allocations and deallocations done by the user application that is being profiled. See: [CUPTI documentation](https://docs.nvidia.com/cuda/cupti/index.html).

> CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE = 1 The memory is pageable. CUPTI_ACTIVITY_MEMORY_KIND_PINNED = 2 The memory is pinned. CUPTI_ACTIVITY_MEMORY_KIND_DEVICE = 3 The memory is on the device. CUPTI_ACTIVITY_MEMORY_KIND_ARRAY = 4 The memory is an array. CUPTI_ACTIVITY_MEMORY_KIND_MANAGED = 5 The memory is managed CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC = 6 The memory is device static CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC = 7 The memory is managed static

There are three graphs shown in nsys GUI timeline. One graph is called the “Memory usage” under each GPU. It is the sum of memory kinds device and array used by that process. The graph increases when memory allocation APIs (such as cudaMalloc, cudaMallocManaged) are called and the graph decreases when memory deallocation APIs (such as cudaFree) are called.



The second graph, titled ‘Managed Memory usage’, shows the managed memory kind, in this case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED, used by that process.



The third graph called “Static Memory usage” is the sum of memory kind device static and managed static used by the process:

- CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC is a static memory allocation. It does not have a context. Since it is static, it is allocated by variable declaration. For example, \_\_device\_\_ int var;

- CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC s static managed allocation. For example, \_\_device\_\_  \_\_managed\_\_ int var; In other words, this is static equivalent of cudaMallocManaged() API.



The pageable and pinned are both host-side memory calls, so we don’t show those on the GUI timeline.

### Unified Memory Transfer Trace

For Nsight Systems Workstation Edition, Unified Memory (also called Managed Memory) transfer trace is enabled automatically in Nsight Systems when CUDA trace is selected. It incurs no overhead in programs that do not perform any Unified Memory transfers. Data is displayed in the Managed Memory area of the timeline:



**HtoD transfer** indicates the CUDA kernel accessed managed memory that was residing on the host, so the kernel execution paused and transferred the data to the device. Heavy traffic here will incur performance penalties in CUDA kernels, so consider using manual cudaMemcpy operations from pinned host memory instead.

**PtoP transfer** indicates the CUDA kernel accessed managed memory that was residing on a different device, so the kernel execution paused and transferred the data to this device. Heavy traffic here will incur performance penalties, so consider using manual cudaMemcpyPeer operations to transfer from other devices’ memory instead. The row showing these events is for the destination device - the source device is shown in the tooltip for each transfer event.

**DtoH transfer** indicates the CPU accessed managed memory that was residing on a CUDA device, so the CPU execution paused and transferred the data to system memory. Heavy traffic here will incur performance penalties in CPU code, so consider using manual cudaMemcpy operations from pinned host memory instead.

Some Unified Memory transfers are highlighted with red to indicate potential performance issues:



Transfers with the following migration causes are highlighted:

- **Coherence**

  Unified Memory migration occurred to guarantee data coherence. SMs (streaming multiprocessors) stop until the migration completes.

- **Eviction**

  Unified Memory migrated to the CPU because it was evicted to make room for another block of memory on the GPU. This happens due to memory overcommitment which is available on Linux with Compute Capability ≥ 6.

#### Unified Memory CPU Page Faults

The Unified Memory CPU page faults feature in Nsight Systems tracks the page faults that occur when CPU code tries to access a memory page that resides on the device.

Note

Collecting Unified Memory CPU page faults can cause overhead of up to 70% in testing. Use this functionality only when needed.



#### Unified Memory GPU Page Faults

The Unified Memory GPU page faults feature in Nsight Systems tracks the page faults that occur when GPU code tries to access a memory page that resides on the host.

Note

Collecting Unified Memory GPU page faults can cause overhead of up to 70% in testing. Use this functionality only when needed.



### CUDA Graph Trace

Nsight Systems is capable of capturing information about CUDA graphs in your application at either the graph or node granularity. This can be set in the CLI using the `--cuda-graph-trace` option, or in the GUI by setting the appropriate drop down.

> 

When CUDA graph trace is set to `graph`, the users sees each graph as one item on the timeline:

> 

When CUDA graph trace is set to `node`, the users sees each graph as a set of nodes on the timeline:

> 

Tracing CUDA graphs at the graph level rather than the tracing the underlying nodes results in significantly less overhead. This option is only available with CUDA driver 515.43 or higher.

### CUDA Event Trace

Nsight Systems is capable of capturing information about CUDA Events (the synchronization mechanism, e.g. cudaEventRecord(), cudaStreamWaitEvent() etc.) in your application. This can be set in the CLI using the `--cuda-event-trace` option, or in the GUI by setting the appropriate drop down.

> 

When CUDA event trace is set to `true`, users can see device-side CUDA event completion markers in CUDA HW timelines:

> 

Additionally, there will be better correlation support among CUDA Event APIs, for example when clicking a cudaEventRecord(), the related calls such cudaEventSynchronize(), cudaStreamWaitEvent() that operate on the same CUDA event object will be highlighted:

> 

However, there are also some limitations with this feature:

- Currently, CUDA Event object created with cudaEventInterprocess flag and/or used in CUDA Graphs are not supported. The support only works under non-IPC and non-CUDA-Graph scenarios.

- Currently, the underlying mechanism for tracing device-side CUDA Event completion is same as CUDA Event’s own timing functionality (i.e. with a CUDA Event object is created without cudaEventDisableTiming flag). The mechanism is known to increase the possibility of false dependencies among seemingly unrelated CUDA Streams. Therefore, if the app’s behavior changes with this feature, consider disabling it.

- This option is only available with CUDA user-mode driver 12.8 or higher.

### CUDA Python Backtrace

Nsight Systems for Arm server (SBSA) platforms and x86 Linux targets, is capable of capturing Python backtrace information when CUDA backtrace is being captured.

To enable CUDA Python backtrace from Nsight Systems:

**CLI** — Set `--python-backtrace=cuda`.

**GUI** — Select the **Collect Python backtrace for selected API calls** checkbox.

> 

Example screenshot:



### CUDA Functions Skipped by Default

In order to decrease the overhead of CUDA API trace, Nsight Systems does not trace short duration CUDA functions with little impact on performance.

You can turn on full CUDA API trace using the CLI or the GUI, but since CUDA trace overhead is on a per API call basis, this will dramatically impact your overhead and may lead to non-representative performance analysis.

**CUDA Runtime API Calls Skipped by Default**

``` cpp
cudaBindSurfaceToArray
cudaBindTexture
cudaBindTexture2D
cudaBindTextureToArray
cudaBindTextureToMipmappedArray
cudaConfigureCall
cudaCreateSurfaceObject
cudaCreateTextureObject
cudaD3D10MapResources
cudaD3D10RegisterResource
cudaD3D10UnmapResources
cudaD3D10UnregisterResource
cudaD3D9MapResources
cudaD3D9MapVertexBuffer
cudaD3D9RegisterResource
cudaD3D9RegisterVertexBuffer
cudaD3D9UnmapResources
cudaD3D9UnmapVertexBuffer
cudaD3D9UnregisterResource
cudaD3D9UnregisterVertexBuffer
cudaDestroySurfaceObject
cudaDestroyTextureObject
cudaDeviceReset
cudaDeviceSynchronize
cudaEGLStreamConsumerAcquireFrame
cudaEGLStreamConsumerConnect
cudaEGLStreamConsumerConnectWithFlags
cudaEGLStreamConsumerDisconnect
cudaEGLStreamConsumerReleaseFrame
cudaEGLStreamConsumerReleaseFrame
cudaEGLStreamProducerConnect
cudaEGLStreamProducerDisconnect
cudaEGLStreamProducerReturnFrame
cudaEventCreate
cudaEventCreateFromEGLSync
cudaEventCreateWithFlags
cudaEventDestroy
cudaEventQuery
cudaEventRecord
cudaEventRecord_ptsz
cudaEventSynchronize
cudaFree
cudaFreeArray
cudaFreeHost
cudaFreeMipmappedArray
cudaGLMapBufferObject
cudaGLMapBufferObjectAsync
cudaGLRegisterBufferObject
cudaGLUnmapBufferObject
cudaGLUnmapBufferObjectAsync
cudaGLUnregisterBufferObject
cudaGraphicsD3D10RegisterResource
cudaGraphicsD3D11RegisterResource
cudaGraphicsD3D9RegisterResource
cudaGraphicsEGLRegisterImage
cudaGraphicsGLRegisterBuffer
cudaGraphicsGLRegisterImage
cudaGraphicsMapResources
cudaGraphicsUnmapResources
cudaGraphicsUnregisterResource
cudaGraphicsVDPAURegisterOutputSurface
cudaGraphicsVDPAURegisterVideoSurface
cudaHostAlloc
cudaHostRegister
cudaHostUnregister
cudaLaunch
cudaLaunchCooperativeKernel
cudaLaunchCooperativeKernelMultiDevice
cudaLaunchCooperativeKernel_ptsz
cudaLaunchKernel
cudaLaunchKernel_ptsz
cudaLaunch_ptsz
cudaMalloc
cudaMalloc3D
cudaMalloc3DArray
cudaMallocArray
cudaMallocHost
cudaMallocManaged
cudaMallocMipmappedArray
cudaMallocPitch
cudaMemGetInfo
cudaMemPrefetchAsync
cudaMemPrefetchAsync_ptsz
cudaMemcpy
cudaMemcpy2D
cudaMemcpy2DArrayToArray
cudaMemcpy2DArrayToArray_ptds
cudaMemcpy2DAsync
cudaMemcpy2DAsync_ptsz
cudaMemcpy2DFromArray
cudaMemcpy2DFromArrayAsync
cudaMemcpy2DFromArrayAsync_ptsz
cudaMemcpy2DFromArray_ptds
cudaMemcpy2DToArray
cudaMemcpy2DToArrayAsync
cudaMemcpy2DToArrayAsync_ptsz
cudaMemcpy2DToArray_ptds
cudaMemcpy2D_ptds
cudaMemcpy3D
cudaMemcpy3DAsync
cudaMemcpy3DAsync_ptsz
cudaMemcpy3DPeer
cudaMemcpy3DPeerAsync
cudaMemcpy3DPeerAsync_ptsz
cudaMemcpy3DPeer_ptds
cudaMemcpy3D_ptds
cudaMemcpyArrayToArray
cudaMemcpyArrayToArray_ptds
cudaMemcpyAsync
cudaMemcpyAsync_ptsz
cudaMemcpyFromArray
cudaMemcpyFromArrayAsync
cudaMemcpyFromArrayAsync_ptsz
cudaMemcpyFromArray_ptds
cudaMemcpyFromSymbol
cudaMemcpyFromSymbolAsync
cudaMemcpyFromSymbolAsync_ptsz
cudaMemcpyFromSymbol_ptds
cudaMemcpyPeer
cudaMemcpyPeerAsync
cudaMemcpyToArray
cudaMemcpyToArrayAsync
cudaMemcpyToArrayAsync_ptsz
cudaMemcpyToArray_ptds
cudaMemcpyToSymbol
cudaMemcpyToSymbolAsync
cudaMemcpyToSymbolAsync_ptsz
cudaMemcpyToSymbol_ptds
cudaMemcpy_ptds
cudaMemset
cudaMemset2D
cudaMemset2DAsync
cudaMemset2DAsync_ptsz
cudaMemset2D_ptds
cudaMemset3D
cudaMemset3DAsync
cudaMemset3DAsync_ptsz
cudaMemset3D_ptds
cudaMemsetAsync
cudaMemsetAsync_ptsz
cudaMemset_ptds
cudaPeerRegister
cudaPeerUnregister
cudaStreamAddCallback
cudaStreamAddCallback_ptsz
cudaStreamAttachMemAsync
cudaStreamAttachMemAsync_ptsz
cudaStreamCreate
cudaStreamCreateWithFlags
cudaStreamCreateWithPriority
cudaStreamDestroy
cudaStreamQuery
cudaStreamQuery_ptsz
cudaStreamSynchronize
cudaStreamSynchronize_ptsz
cudaStreamWaitEvent
cudaStreamWaitEvent_ptsz
cudaThreadSynchronize
cudaUnbindTexture
```

**CUDA Primary API Calls Skipped by Default**

``` cpp
cu64Array3DCreate
cu64ArrayCreate
cu64D3D9MapVertexBuffer
cu64GLMapBufferObject
cu64GLMapBufferObjectAsync
cu64MemAlloc
cu64MemAllocPitch
cu64MemFree
cu64MemGetInfo
cu64MemHostAlloc
cu64Memcpy2D
cu64Memcpy2DAsync
cu64Memcpy2DUnaligned
cu64Memcpy3D
cu64Memcpy3DAsync
cu64MemcpyAtoD
cu64MemcpyDtoA
cu64MemcpyDtoD
cu64MemcpyDtoDAsync
cu64MemcpyDtoH
cu64MemcpyDtoHAsync
cu64MemcpyHtoD
cu64MemcpyHtoDAsync
cu64MemsetD16
cu64MemsetD16Async
cu64MemsetD2D16
cu64MemsetD2D16Async
cu64MemsetD2D32
cu64MemsetD2D32Async
cu64MemsetD2D8
cu64MemsetD2D8Async
cu64MemsetD32
cu64MemsetD32Async
cu64MemsetD8
cu64MemsetD8Async
cuArray3DCreate
cuArray3DCreate_v2
cuArrayCreate
cuArrayCreate_v2
cuArrayDestroy
cuBinaryFree
cuCompilePtx
cuCtxCreate
cuCtxCreate_v2
cuCtxDestroy
cuCtxDestroy_v2
cuCtxSynchronize
cuD3D10CtxCreate
cuD3D10CtxCreateOnDevice
cuD3D10CtxCreate_v2
cuD3D10MapResources
cuD3D10RegisterResource
cuD3D10UnmapResources
cuD3D10UnregisterResource
cuD3D11CtxCreate
cuD3D11CtxCreateOnDevice
cuD3D11CtxCreate_v2
cuD3D9CtxCreate
cuD3D9CtxCreateOnDevice
cuD3D9CtxCreate_v2
cuD3D9MapResources
cuD3D9MapVertexBuffer
cuD3D9MapVertexBuffer_v2
cuD3D9RegisterResource
cuD3D9RegisterVertexBuffer
cuD3D9UnmapResources
cuD3D9UnmapVertexBuffer
cuD3D9UnregisterResource
cuD3D9UnregisterVertexBuffer
cuEGLStreamConsumerAcquireFrame
cuEGLStreamConsumerConnect
cuEGLStreamConsumerConnectWithFlags
cuEGLStreamConsumerDisconnect
cuEGLStreamConsumerReleaseFrame
cuEGLStreamProducerConnect
cuEGLStreamProducerDisconnect
cuEGLStreamProducerPresentFrame
cuEGLStreamProducerReturnFrame
cuEventCreate
cuEventCreateFromEGLSync
cuEventCreateFromNVNSync
cuEventDestroy
cuEventDestroy_v2
cuEventQuery
cuEventRecord
cuEventRecord_ptsz
cuEventSynchronize
cuGLCtxCreate
cuGLCtxCreate_v2
cuGLInit
cuGLMapBufferObject
cuGLMapBufferObjectAsync
cuGLMapBufferObjectAsync_v2
cuGLMapBufferObjectAsync_v2_ptsz
cuGLMapBufferObject_v2
cuGLMapBufferObject_v2_ptds
cuGLRegisterBufferObject
cuGLUnmapBufferObject
cuGLUnmapBufferObjectAsync
cuGLUnregisterBufferObject
cuGraphicsD3D10RegisterResource
cuGraphicsD3D11RegisterResource
cuGraphicsD3D9RegisterResource
cuGraphicsEGLRegisterImage
cuGraphicsGLRegisterBuffer
cuGraphicsGLRegisterImage
cuGraphicsMapResources
cuGraphicsMapResources_ptsz
cuGraphicsUnmapResources
cuGraphicsUnmapResources_ptsz
cuGraphicsUnregisterResource
cuGraphicsVDPAURegisterOutputSurface
cuGraphicsVDPAURegisterVideoSurface
cuInit
cuLaunch
cuLaunchCooperativeKernel
cuLaunchCooperativeKernelMultiDevice
cuLaunchCooperativeKernel_ptsz
cuLaunchGrid
cuLaunchGridAsync
cuLaunchKernel
cuLaunchKernel_ptsz
cuLinkComplete
cuLinkCreate
cuLinkCreate_v2
cuLinkDestroy
cuMemAlloc
cuMemAllocHost
cuMemAllocHost_v2
cuMemAllocManaged
cuMemAllocPitch
cuMemAllocPitch_v2
cuMemAlloc_v2
cuMemFree
cuMemFreeHost
cuMemFree_v2
cuMemGetInfo
cuMemGetInfo_v2
cuMemHostAlloc
cuMemHostAlloc_v2
cuMemHostRegister
cuMemHostRegister_v2
cuMemHostUnregister
cuMemPeerRegister
cuMemPeerUnregister
cuMemPrefetchAsync
cuMemPrefetchAsync_ptsz
cuMemcpy
cuMemcpy2D
cuMemcpy2DAsync
cuMemcpy2DAsync_v2
cuMemcpy2DAsync_v2_ptsz
cuMemcpy2DUnaligned
cuMemcpy2DUnaligned_v2
cuMemcpy2DUnaligned_v2_ptds
cuMemcpy2D_v2
cuMemcpy2D_v2_ptds
cuMemcpy3D
cuMemcpy3DAsync
cuMemcpy3DAsync_v2
cuMemcpy3DAsync_v2_ptsz
cuMemcpy3DPeer
cuMemcpy3DPeerAsync
cuMemcpy3DPeerAsync_ptsz
cuMemcpy3DPeer_ptds
cuMemcpy3D_v2
cuMemcpy3D_v2_ptds
cuMemcpyAsync
cuMemcpyAsync_ptsz
cuMemcpyAtoA
cuMemcpyAtoA_v2
cuMemcpyAtoA_v2_ptds
cuMemcpyAtoD
cuMemcpyAtoD_v2
cuMemcpyAtoD_v2_ptds
cuMemcpyAtoH
cuMemcpyAtoHAsync
cuMemcpyAtoHAsync_v2
cuMemcpyAtoHAsync_v2_ptsz
cuMemcpyAtoH_v2
cuMemcpyAtoH_v2_ptds
cuMemcpyDtoA
cuMemcpyDtoA_v2
cuMemcpyDtoA_v2_ptds
cuMemcpyDtoD
cuMemcpyDtoDAsync
cuMemcpyDtoDAsync_v2
cuMemcpyDtoDAsync_v2_ptsz
cuMemcpyDtoD_v2
cuMemcpyDtoD_v2_ptds
cuMemcpyDtoH
cuMemcpyDtoHAsync
cuMemcpyDtoHAsync_v2
cuMemcpyDtoHAsync_v2_ptsz
cuMemcpyDtoH_v2
cuMemcpyDtoH_v2_ptds
cuMemcpyHtoA
cuMemcpyHtoAAsync
cuMemcpyHtoAAsync_v2
cuMemcpyHtoAAsync_v2_ptsz
cuMemcpyHtoA_v2
cuMemcpyHtoA_v2_ptds
cuMemcpyHtoD
cuMemcpyHtoDAsync
cuMemcpyHtoDAsync_v2
cuMemcpyHtoDAsync_v2_ptsz
cuMemcpyHtoD_v2
cuMemcpyHtoD_v2_ptds
cuMemcpyPeer
cuMemcpyPeerAsync
cuMemcpyPeerAsync_ptsz
cuMemcpyPeer_ptds
cuMemcpy_ptds
cuMemcpy_v2
cuMemsetD16
cuMemsetD16Async
cuMemsetD16Async_ptsz
cuMemsetD16_v2
cuMemsetD16_v2_ptds
cuMemsetD2D16
cuMemsetD2D16Async
cuMemsetD2D16Async_ptsz
cuMemsetD2D16_v2
cuMemsetD2D16_v2_ptds
cuMemsetD2D32
cuMemsetD2D32Async
cuMemsetD2D32Async_ptsz
cuMemsetD2D32_v2
cuMemsetD2D32_v2_ptds
cuMemsetD2D8
cuMemsetD2D8Async
cuMemsetD2D8Async_ptsz
cuMemsetD2D8_v2
cuMemsetD2D8_v2_ptds
cuMemsetD32
cuMemsetD32Async
cuMemsetD32Async_ptsz
cuMemsetD32_v2
cuMemsetD32_v2_ptds
cuMemsetD8
cuMemsetD8Async
cuMemsetD8Async_ptsz
cuMemsetD8_v2
cuMemsetD8_v2_ptds
cuMipmappedArrayCreate
cuMipmappedArrayDestroy
cuModuleLoad
cuModuleLoadData
cuModuleLoadDataEx
cuModuleLoadFatBinary
cuModuleUnload
cuStreamAddCallback
cuStreamAddCallback_ptsz
cuStreamAttachMemAsync
cuStreamAttachMemAsync_ptsz
cuStreamBatchMemOp
cuStreamBatchMemOp_ptsz
cuStreamCreate
cuStreamCreateWithPriority
cuStreamDestroy
cuStreamDestroy_v2
cuStreamSynchronize
cuStreamSynchronize_ptsz
cuStreamWaitEvent
cuStreamWaitEvent_ptsz
cuStreamWaitValue32
cuStreamWaitValue32_ptsz
cuStreamWaitValue64
cuStreamWaitValue64_ptsz
cuStreamWriteValue32
cuStreamWriteValue32_ptsz
cuStreamWriteValue64
cuStreamWriteValue64_ptsz
cuSurfObjectCreate
cuSurfObjectDestroy
cuSurfRefCreate
cuSurfRefDestroy
cuTexObjectCreate
cuTexObjectDestroy
cuTexRefCreate
cuTexRefDestroy
cuVDPAUCtxCreate
cuVDPAUCtxCreate_v2
```

### cuDNN Function List for X86 CLI

**cuDNN API functions**

``` cpp
cudnnActivationBackward
cudnnActivationBackward_v3
cudnnActivationBackward_v4
cudnnActivationForward
cudnnActivationForward_v3
cudnnActivationForward_v4
cudnnAddTensor
cudnnBatchNormalizationBackward
cudnnBatchNormalizationBackwardEx
cudnnBatchNormalizationForwardInference
cudnnBatchNormalizationForwardTraining
cudnnBatchNormalizationForwardTrainingEx
cudnnCTCLoss
cudnnConvolutionBackwardBias
cudnnConvolutionBackwardData
cudnnConvolutionBackwardFilter
cudnnConvolutionBiasActivationForward
cudnnConvolutionForward
cudnnCreate
cudnnCreateAlgorithmPerformance
cudnnDestroy
cudnnDestroyAlgorithmPerformance
cudnnDestroyPersistentRNNPlan
cudnnDivisiveNormalizationBackward
cudnnDivisiveNormalizationForward
cudnnDropoutBackward
cudnnDropoutForward
cudnnDropoutGetReserveSpaceSize
cudnnDropoutGetStatesSize
cudnnFindConvolutionBackwardDataAlgorithm
cudnnFindConvolutionBackwardDataAlgorithmEx
cudnnFindConvolutionBackwardFilterAlgorithm
cudnnFindConvolutionBackwardFilterAlgorithmEx
cudnnFindConvolutionForwardAlgorithm
cudnnFindConvolutionForwardAlgorithmEx
cudnnFindRNNBackwardDataAlgorithmEx
cudnnFindRNNBackwardWeightsAlgorithmEx
cudnnFindRNNForwardInferenceAlgorithmEx
cudnnFindRNNForwardTrainingAlgorithmEx
cudnnFusedOpsExecute
cudnnIm2Col
cudnnLRNCrossChannelBackward
cudnnLRNCrossChannelForward
cudnnMakeFusedOpsPlan
cudnnMultiHeadAttnBackwardData
cudnnMultiHeadAttnBackwardWeights
cudnnMultiHeadAttnForward
cudnnOpTensor
cudnnPoolingBackward
cudnnPoolingForward
cudnnRNNBackwardData
cudnnRNNBackwardDataEx
cudnnRNNBackwardWeights
cudnnRNNBackwardWeightsEx
cudnnRNNForwardInference
cudnnRNNForwardInferenceEx
cudnnRNNForwardTraining
cudnnRNNForwardTrainingEx
cudnnReduceTensor
cudnnReorderFilterAndBias
cudnnRestoreAlgorithm
cudnnRestoreDropoutDescriptor
cudnnSaveAlgorithm
cudnnScaleTensor
cudnnSoftmaxBackward
cudnnSoftmaxForward
cudnnSpatialTfGridGeneratorBackward
cudnnSpatialTfGridGeneratorForward
cudnnSpatialTfSamplerBackward
cudnnSpatialTfSamplerForward
cudnnTransformFilter
cudnnTransformTensor
cudnnTransformTensorEx
```

## OpenMP Trace

Nsight Systems for Linux is capable of capturing information about OpenMP events. This functionality is built on the OpenMP Tools Interface (OMPT), full support is available only for runtime libraries supporting tools interface defined in OpenMP 5.0 or greater.

As an example, LLVM OpenMP runtime library partially implements tools interface. If you use PGI compiler \<= 20.4 to build your OpenMP applications, add the `-mp=libomp` switch to use LLVM OpenMP runtime and enable OMPT based tracing. If you use Clang, make sure the LLVM OpenMP runtime library you link to was compiled with tools interface enabled.



Only a subset of the OMPT callbacks are processed:

``` cpp
ompt_callback_parallel_begin
ompt_callback_parallel_end
ompt_callback_sync_region
ompt_callback_task_create
ompt_callback_task_schedule
ompt_callback_implicit_task
ompt_callback_master
ompt_callback_reduction
ompt_callback_task_create
ompt_callback_cancel
ompt_callback_mutex_acquire, ompt_callback_mutex_acquired
ompt_callback_mutex_acquired, ompt_callback_mutex_released
ompt_callback_mutex_released
ompt_callback_work
ompt_callback_dispatch
ompt_callback_flush
```

Note

The raw OMPT events are used to generate ranges indicating the runtime of OpenMP operations and constructs.

Example screenshot:



## OS Runtime Libraries Trace

On Linux, OS runtime libraries can be traced to gather information about low-level userspace APIs. This traces the system call wrappers and thread synchronization interfaces exposed by the C runtime and POSIX Threads (pthread) libraries. This does not perform a complete runtime library API trace, but instead focuses on the functions that can take a long time to execute, or could potentially cause your thread be unscheduled from the CPU while waiting for an event to complete. OS runtime trace is not available for Windows targets.

OS runtime tracing complements and enhances sampling information by:

1.  Visualizing when the process is communicating with the hardware, controlling resources, performing multi-threading synchronization or interacting with the kernel scheduler.

2.  Adding additional thread states by correlating how OS runtime libraries traces affect the thread scheduling:

    - **Waiting** — the thread is not scheduled on a CPU, it is inside of an OS runtime libraries trace and is believed to be waiting on the firmware to complete a request.

    - **In OS runtime library function** — the thread is scheduled on a CPU and inside of an OS runtime libraries trace. If the trace represents a system call, the process is likely running in kernel mode.

3.  Collecting backtraces for long OS runtime libraries call. This provides a way to gather blocked-state backtraces, allowing you to gain more context about why the thread was blocked so long, yet avoiding unnecessary overhead for short events.

    

4.  Collecting file access data for API calls that interact with files. This helps in identifying performance bottlenecks related to file I/O operations and provides insights into how file access patterns affect overall application performance.

    >  

    Note

    File access data collection is not enabled by default.

To enable OS runtime libraries tracing from Nsight Systems:

**CLI** — Use the `-t`, `--trace` option with the `osrt` parameter. See [Command Line Options](index.html#command-line-options) for more information.

**GUI** — Select the **Collect OS runtime libraries trace** checkbox.

> 

You can also use **Skip if shorter than**. This will skip calls shorter than the given threshold. Enabling this option will improve performances as well as reduce noise on the timeline. We strongly encourage you to skip OS runtime libraries call shorter than 1 μs.

### Locking a Resource

The functions listed below receive a special treatment. If the tool detects that the resource is already acquired by another thread and will induce a blocking call, we always trace it. Otherwise, it will never be traced.

``` cpp
pthread_mutex_lock
pthread_rwlock_rdlock
pthread_rwlock_wrlock
pthread_spin_lock
sem_wait
```

Note that even if a call is determined as potentially blocking, there is a chance that it may not actually block after a few cycles have elapsed. The call will still be traced in this scenario.

### Limitations

- Nsight Systems only traces syscall wrappers exposed by the C runtime. It is not able to trace syscall invoked through assembly code.

- Additional thread states, as well as backtrace collection on long calls, are only enabled if sampling is turned on.

- It is not possible to configure the depth and duration threshold when collecting backtraces. Currently, only OS runtime libraries calls longer than 80 μs will generate a backtrace with a maximum of 24 frames. This limitation will be removed in a future version of the product.

- It is required to compile your application and libraries with the `-funwind-tables` compiler flag in order for Nsight Systems to unwind the backtraces correctly.

### OS Runtime Libraries Trace Filters

The OS runtime libraries tracing is limited to a select list of functions. It also depends on the version of the C runtime linked to the application.

### OS Runtime Default Function List

**Libc system call wrappers**

``` cpp
accept
accept4
acct
alarm
arch_prctl
bind
bpf
brk
chroot
clock_nanosleep
connect
copy_file_range
creat
creat64
dup
dup2
dup3
epoll_ctl
epoll_pwait
epoll_wait
fallocate
fallocate64
fcntl
fdatasync
flock
fork
fsync
ftruncate
futex
ioctl
ioperm
iopl
kill
killpg
listen
membarrier
mlock
mlock2
mlockall
mmap
mmap64
mount
move_pages
mprotect
mq_notify
mq_open
mq_receive
mq_send
mq_timedreceive
mq_timedsend
mremap
msgctl
msgget
msgrcv
msgsnd
msync
munmap
nanosleep
nfsservctl
open
open64
openat
openat64
pause
pipe
pipe2
pivot_root
poll
ppoll
prctl
pread
pread64
preadv
preadv2
preadv64
process_vm_readv
process_vm_writev
pselect6
ptrace
pwrite
pwrite64
pwritev
pwritev2
pwritev64
read
readv
reboot
recv
recvfrom
recvmmsg
recvmsg
rt_sigaction
rt_sigqueueinfo
rt_sigsuspend
rt_sigtimedwait
sched_yield
seccomp
select
semctl
semget
semop
semtimedop
send
sendfile
sendfile64
sendmmsg
sendmsg
sendto
shmat
shmctl
shmdt
shmget
shutdown
sigaction
sigsuspend
sigtimedwait
socket
socketpair
splice
swapoff
swapon
sync
sync_file_range
syncfs
tee
tgkill
tgsigqueueinfo
tkill
truncate
umount2
unshare
uselib
vfork
vhangup
vmsplice
wait
wait3
wait4
waitid
waitpid
write
writev
_sysctl
```

**POSIX Threads**

``` cpp
pthread_barrier_wait
pthread_cancel
pthread_cond_broadcast
pthread_cond_signal
pthread_cond_timedwait
pthread_cond_wait
pthread_create
pthread_join
pthread_kill
pthread_mutex_lock
pthread_mutex_timedlock
pthread_mutex_trylock
pthread_rwlock_rdlock
pthread_rwlock_timedrdlock
pthread_rwlock_timedwrlock
pthread_rwlock_tryrdlock
pthread_rwlock_trywrlock
pthread_rwlock_wrlock
pthread_spin_lock
pthread_spin_trylock
pthread_timedjoin_np
pthread_tryjoin_np
pthread_yield
sem_timedwait
sem_trywait
sem_wait
```

**I/O**

``` cpp
aio_fsync
aio_fsync64
aio_suspend
aio_suspend64
fclose
fcloseall
fflush
fflush_unlocked
fgetc
fgetc_unlocked
fgets
fgets_unlocked
fgetwc
fgetwc_unlocked
fgetws
fgetws_unlocked
flockfile
fopen
fopen64
fputc
fputc_unlocked
fputs
fputs_unlocked
fputwc
fputwc_unlocked
fputws
fputws_unlocked
fread
fread_unlocked
freopen
freopen64
ftrylockfile
fwrite
fwrite_unlocked
getc
getc_unlocked
getdelim
getline
getw
getwc
getwc_unlocked
lockf
lockf64
mkfifo
mkfifoat
posix_fallocate
posix_fallocate64
putc
putc_unlocked
putwc
putwc_unlocked
```

**Miscellaneous**

``` cpp
forkpty
popen
posix_spawn
posix_spawnp
sigwait
sigwaitinfo
sleep
system
usleep
```

## Syscall Trace

Nsight Systems for Linux and Nsight Systems Embedded Platforms Edition are capable of tracing Linux system calls in kernel space. This feature uses Linux’s eBPF technology, and is supported on systems running Linux v5.11 or newer, specifically those that are built with `CONFIG_DEBUG_INFO_BTF` enabled, which is the default on most major Linux distros. This feature requires `CAP_BPF` and `CAP_PERFMON` capabilities (alternatively, `CAP_SYS_ADMIN` or root privileges) for the `nsys` process, see the capabilities [man page](https://man7.org/linux/man-pages/man7/capabilities.7.html) for details.

To enable this feature:

**CLI** — Add the `--syscall` option to the `nsys`` ``start` or `nsys`` ``profile` commands (setting `syscall` in the `--trace` option is deprecated and will be ignored). The following values are supported:

- `none` — No syscall tracing \[default\].

- `process-tree` — Collect syscalls for the profiled application process and its child processes.

- `pid-namespace` — Collect syscalls made by all processes in the current PID namespace and its child namespaces. This is very close to how other features work in the `system-wide` mode, e.g. inside a container, tracing will be limited to this container.

**GUI** — Select the **Collect syscall trace** checkbox. Currently, equivalent to the `--syscall=process-tree` option.

> 

Please note that only syscalls running 1000ns and more are traced.

Example screenshot:

> 

Long running (more than 80 microseconds) syscalls are also collected with their backtraces:

> 

## NVTX Trace

The NVIDIA Tools Extension Library (NVTX) is a powerful mechanism that allows users to manually instrument their application. Nsight Systems can then collect the information and present it on the timeline.

NVTX is shipped as a C-based, header-only library. NVIDIA also supports and provides library-specific wrappers for C++, Python, and Rust. See the [NVTX GitHub repository](https://nvidia.github.io/NVTX/).

Nsight Systems supports version 3.0 of the NVTX specification.

The most commonly used features:

- Domains

  ``` cpp
  nvtxDomainCreate(), nvtxDomainDestroy()
  ```

  ``` cpp
  nvtxDomainRegisterString()
  ```

- Push-pop ranges (nested ranges that start and end in the same thread).

  ``` cpp
  nvtxRangePush(), nvtxRangePushEx()
  ```

  ``` cpp
  nvtxRangePop()
  ```

  ``` cpp
  nvtxDomainRangePushEx()
  ```

  ``` cpp
  nvtxDomainRangePop()
  ```

- Start-end ranges (ranges that are global to the process and are not restricted to a single thread)

  ``` cpp
  nvtxRangeStart(), nvtxRangeStartEx()
  ```

  ``` cpp
  nvtxRangeEnd()
  ```

  ``` cpp
  nvtxDomainRangeStartEx()
  ```

  ``` cpp
  nvtxDomainRangeEnd()
  ```

- Marks

  ``` cpp
  nvtxMark(), nvtxMarkEx()
  ```

  ``` cpp
  nvtxDomainMarkEx()
  ```

- Thread names

  ``` cpp
  nvtxNameOsThread()
  ```

- Categories

  ``` cpp
  nvtxNameCategory()
  ```

  ``` cpp
  nvtxDomainNameCategory()
  ```

To learn more about specific features of NVTX, please refer to the NVTX header file: `nvToolsExt.h` or the [NVTX documentation](https://nvidia.github.io/NVTX/).

Note

It is strongly recommended that you use registered strings for your range names. This enables profiling tools to use a more performant match algorithm. For more information about creating registered strings, see [NVTX String Registration](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx-string-registration).

To use NVTX in your application, follow these steps:

1.  Add `#include`` ``"nvtx3/nvToolsExt.h"` in your source code. The nvtx3 directory is located in the Nsight Systems package in the `Target-<architecture>/nvtx/include` directory and is available via github at [NVIDIA/NVTX](http://github.com/NVIDIA/NVTX).

2.  Add the following compiler flag: `-ldl`

3.  Add calls to the NVTX API functions. For example, try adding `nvtxRangePush("main")` in the beginning of the `main()` function, and `nvtxRangePop()` just before the return statement in the end.

    For convenience in C++ code, consider adding a wrapper that implements RAII (resource acquisition is initialization) pattern, which would guarantee that every range gets closed.

4.  In the project settings, select the **Collect NVTX trace** checkbox.

In addition, by enabling the “Insert NVTX Marker hotkey” option it is possible to add NVTX markers to a running non-console applications by pressing the F11 key. These will appear in the report under the NVTX Domain named “HotKey markers.”

Typically, calls to NVTX functions can be left in the source code even if the application is not being built for profiling purposes, since the overhead is very low when the profiler is not attached.

NVTX is not intended to annotate very small pieces of code that are being called very frequently. A good rule of thumb to use: if code being annotated usually takes less than 1 microsecond to execute, adding an NVTX range around this code should be done carefully.

Note

Range annotations should be matched carefully. If many ranges are opened but not closed, Nsight Systems has no meaningful way to visualize it. A rule of thumb is to not have more than a couple dozen ranges open at any point in time. Nsight Systems does not support reports with many unclosed ranges.

**Sample with C++ Wrapper**

The following example uses the NVTX C++ wrapper to create a range around `some_function()` and a nested range for each loop iteration.

``` cpp
#include <nvtx3/nvtx3.hpp>
#include <thread>
#include <chrono>

void some_function()
{
    NVTX3_FUNC_RANGE();  // Range around the whole function

    for (int i = 0; i < 6; ++i) {
        nvtx3::scoped_range loop{"loop range"};  // Range for iteration

        // Make each iteration last for one second
        std::this_thread::sleep_for(std::chrono::seconds{1});
    }
}
```

A complete program that calls `some_function()` does not require linking to an NVTX library. Compile the program as usual.

``` cpp
g++ -o example example.cpp
```

Run the executable with `nsys` to collect and view the data.

``` cpp
nsys profile ./example
nsys-ui report1.nsys-rep
```

The NVTX C++ header is available in the `<target-platform-folder>/nvtx/include` directory of the Nsight Systems installation.

``` cpp
export NSYS_NVTX_PATH=<nsys_install_dir>/<target-platform-folder>/nvtx
g++ -o example example.cpp -I${NSYS_NVTX_PATH}/include
```

Using the NVTX C API, the following example creates the same function and loop ranges.

``` cpp
#include <nvtx3/nvToolsExt.h>
#include <thread>
#include <chrono>

void some_function()
{
    nvtxRangePush(__func__);

    for (int i = 0; i < 6; ++i) {
        nvtxRangePush("loop range");

        std::this_thread::sleep_for(std::chrono::seconds{1});

        nvtxRangePop();
    }

    nvtxRangePop();
}
```



The NVTX row shows the function’s name “some_function” in the top-level range and the “loop range” message in the nested ranges. The loop iterations each last for the expected one second.

If the function returns or throws before calling `nvtxRangePop`, the range is left unclosed and tool behavior is undefined. The C++ API is safer because the range object ends the range from its destructor.

**NVTX Payloads and Counters**

NVTX Extended Payloads and NVTX Counters increase the flexibility of NVTX annotations by allowing users to pass arbitrary structured data to NVTX events. This then will allow users to define the layout of this data in the Nsight Systems UI without additional data conversion.

For more information, see [NVTX documentation](https://nvidia.github.io/NVTX/).

> 

**NVTX Domains and Categories**

NVTX domains enable scoping of annotations. Unless specified differently, all events and annotations are in the default domain. Additionally, categories can be used to group events.

Nsight Systems gives the user the ability to include or exclude NVTX events from a particular domain. This can be especially useful if you are profiling across multiple libraries and are only interested in nvtx events from some of them.



This functionality is also available from the CLI. See the CLI documentation for `--nvtx-domain-include` and `--nvtx-domain-exclude` for more details.

Categories that are set in by the user will be recognized and displayed in the GUI.



## OpenACC Trace

Nsight Systems for Linux x86_64 is capable of capturing information about OpenACC execution in the profiled process.

OpenACC versions 2.0, 2.5, and 2.6 are supported when using PGI runtime version 15.7 or later. In order to differentiate constructs (see tooltip below), a PGI runtime of 16.0 or later is required. Note that Nsight Systems does not support the GCC implementation of OpenACC at this time.

Under the CPU rows in the timeline tree, each thread that uses OpenACC will show OpenACC trace information. You can click on an OpenACC API call to see correlation with the underlying CUDA API calls (highlighted in teal):



If the OpenACC API results in GPU work, that will also be highlighted:



Hovering over a particular OpenACC construct will bring up a tooltip with details about that construct:

> 

To capture OpenACC information from the Nsight Systems GUI, select the **Collect OpenACC trace** checkbox under **Collect CUDA trace** configurations. Note that turning on OpenACC tracing will also turn on CUDA tracing.

> 

Note

If your application crashes before all collected OpenACC trace data has been copied out, some or all data might be lost and not present in the report.

## OpenGL Trace

OpenGL and OpenGL ES APIs can be traced to assist in the analysis of CPU and GPU interactions.

A few usage examples are:

1.  Visualize how long `eglSwapBuffers` (or similar) is taking.

2.  API trace can easily show correlations between thread state and graphics driver’s behavior, uncovering where the CPU may be waiting on the GPU.

3.  Spot bubbles of opportunity on the GPU, where more GPU workload could be created.

4.  Use `KHR_debug` extension to trace GL events on both the CPU and GPU.

OpenGL trace feature in Nsight Systems consists of two different activities which will be shown in the CPU rows for those threads

- **CPU trace**: interception of API calls that an application does to APIs (such as OpenGL, OpenGL ES, EGL, GLX, WGL, etc.).

- **GPU trace** (or **workload trace**): trace of GPU workload (activity) triggered by use of OpenGL or OpenGL ES. Since draw calls are executed back-to-back, the GPU workload trace ranges include many OpenGL draw calls and operations in order to optimize performance overhead, rather than tracing each individual operation.

To collect GPU trace, the `glQueryCounter()` function is used to measure how much time batches of GPU workload take to complete.

> 

> 

Ranges defined by the `KHR_debug` calls are represented similarly to OpenGL API and OpenGL GPU workload trace. GPU ranges in this case represent *incremental draw cost*. They cannot fully account for GPUs that can execute multiple draw calls in parallel. In this case, Nsight Systems will not show overlapping GPU ranges.

### OpenGL Trace Using Command Line

For general information on using the target CLI, see [CLI Profiling on Linux](index.html#cli-profiling). For the CLI, the functions that are traced are set to the following list:

``` cpp
glWaitSync
glReadPixels
glReadnPixelsKHR
glReadnPixelsEXT
glReadnPixelsARB
glReadnPixels
glFlush
glFinishFenceNV
glFinish
glClientWaitSync
glClearTexSubImage
glClearTexImage
glClearStencil
glClearNamedFramebufferuiv
glClearNamedFramebufferiv
glClearNamedFramebufferfv
glClearNamedFramebufferfi
glClearNamedBufferSubDataEXT
glClearNamedBufferSubData
glClearNamedBufferDataEXT
glClearNamedBufferData
glClearIndex
glClearDepthx
glClearDepthf
glClearDepthdNV
glClearDepth
glClearColorx
glClearColorIuiEXT
glClearColorIiEXT
glClearColor
glClearBufferuiv
glClearBufferSubData
glClearBufferiv
glClearBufferfv
glClearBufferfi
glClearBufferData
glClearAccum
glClear
glDispatchComputeIndirect
glDispatchComputeGroupSizeARB
glDispatchCompute
glComputeStreamNV
glNamedFramebufferDrawBuffers
glNamedFramebufferDrawBuffer
glMultiDrawElementsIndirectEXT
glMultiDrawElementsIndirectCountARB
glMultiDrawElementsIndirectBindlessNV
glMultiDrawElementsIndirectBindlessCountNV
glMultiDrawElementsIndirectAMD
glMultiDrawElementsIndirect
glMultiDrawElementsEXT
glMultiDrawElementsBaseVertex
glMultiDrawElements
glMultiDrawArraysIndirectEXT
glMultiDrawArraysIndirectCountARB
glMultiDrawArraysIndirectBindlessNV
glMultiDrawArraysIndirectBindlessCountNV
glMultiDrawArraysIndirectAMD
glMultiDrawArraysIndirect
glMultiDrawArraysEXT
glMultiDrawArrays
glListDrawCommandsStatesClientNV
glFramebufferDrawBuffersEXT
glFramebufferDrawBufferEXT
glDrawTransformFeedbackStreamInstanced
glDrawTransformFeedbackStream
glDrawTransformFeedbackNV
glDrawTransformFeedbackInstancedEXT
glDrawTransformFeedbackInstanced
glDrawTransformFeedbackEXT
glDrawTransformFeedback
glDrawTexxvOES
glDrawTexxOES
glDrawTextureNV
glDrawTexsvOES
glDrawTexsOES
glDrawTexivOES
glDrawTexiOES
glDrawTexfvOES
glDrawTexfOES
glDrawRangeElementsEXT
glDrawRangeElementsBaseVertexOES
glDrawRangeElementsBaseVertexEXT
glDrawRangeElementsBaseVertex
glDrawRangeElements
glDrawPixels
glDrawElementsInstancedNV
glDrawElementsInstancedEXT
glDrawElementsInstancedBaseVertexOES
glDrawElementsInstancedBaseVertexEXT
glDrawElementsInstancedBaseVertexBaseInstanceEXT
glDrawElementsInstancedBaseVertexBaseInstance
glDrawElementsInstancedBaseVertex
glDrawElementsInstancedBaseInstanceEXT
glDrawElementsInstancedBaseInstance
glDrawElementsInstancedARB
glDrawElementsInstanced
glDrawElementsIndirect
glDrawElementsBaseVertexOES
glDrawElementsBaseVertexEXT
glDrawElementsBaseVertex
glDrawElements
glDrawCommandsStatesNV
glDrawCommandsStatesAddressNV
glDrawCommandsNV
glDrawCommandsAddressNV
glDrawBuffersNV
glDrawBuffersATI
glDrawBuffersARB
glDrawBuffers
glDrawBuffer
glDrawArraysInstancedNV
glDrawArraysInstancedEXT
glDrawArraysInstancedBaseInstanceEXT
glDrawArraysInstancedBaseInstance
glDrawArraysInstancedARB
glDrawArraysInstanced
glDrawArraysIndirect
glDrawArraysEXT
glDrawArrays
eglSwapBuffersWithDamageKHR
eglSwapBuffers
glXSwapBuffers
glXQueryDrawable
glXGetCurrentReadDrawable
glXGetCurrentDrawable
glGetQueryObjectuivEXT
glGetQueryObjectuivARB
glGetQueryObjectuiv
glGetQueryObjectivARB
glGetQueryObjectiv
```

## OpenXR API Trace

OpenXR is a royalty-free, open standard that provides high-performance access to Augmented Reality (AR) and Virtual Reality (VR)—collectively known as XR—platforms and devices. Information about OpenXR can be found at the [OpenXR Overview](https://www.khronos.org/openxr).

Nsight Systems can capture information about OpenXR usage by the profiled process. This includes capturing the execution time of OpenXR API functions, debug labels, and frame durations. OpenXR profiling is supported on Windows operating systems.



## GPU Hardware Profiling

### GPU Context Switch

Nsight Systems provides the ability to trace GPU context switches.

To enable trace, run from the CLI using the `--gpuctxsw` option

From the GUI:



Specifically, the behavior is as follows:

When collecting GPU context switch data as root, you will get records about contexts from all processes. The records have valid context IDs and process IDs, and have full-precision timestamps.

When collecting GPU context switch data as a normal user, you will still get records about contexts from all processes. For processes running as your user, the records have valid context ID and process IDs, and full-precision timestamps. For processes running as a different user, the records have context ID = 0 and process ID = 0, and reduced-precision timestamps (which are still guaranteed to be in the correct order).

When collecting GPU context switch data in a virtual machine using vGPU, the above rules apply to records relating to your VM. No records are collected for contexts running on other VMs, so the timeline may show gaps when the vGPU is switched to another VM’s context(s). We do not currently support collecting GPU context switch data on a host system where vGPUs are in use by VMs.



### GPU Metrics

#### Overview

GPU Metrics feature is intended to identify performance limiters in applications using GPU for computations and graphics. It uses periodic sampling to gather performance metrics and detailed timing statistics associated with different GPU hardware units taking advantage of specialized hardware to capture this data in a single pass with minimal overhead.

Note

GPU Metrics will give you precise device level information, but it does not know which process or context is involved. GPU context switch trace provides less precise information, but will give you process and context information.



These metrics provide an overview of GPU efficiency over time within compute, graphics, and input/output (IO) activities such as:

- **IO throughputs:** PCIe, NVLink, and GPU memory bandwidth

- **SM utilization:** SMs activity, tensor core activity, instructions issued, warp occupancy, and unassigned warp slots

It is designed to help users answer the common questions:

- Is my GPU idle?

- Is my GPU full? Enough kernel grids size and streams? Are my SMs and warp slots full?

- Am I using TensorCores?

- Is my instruction rate high?

- Am I possibly blocked on IO, or number of warps, etc.?

Nsight Systems GPU Metrics is only available for Linux targets on x86-64 and aarch64, and for Windows targets. It requires NVIDIA Turing architecture or newer.

Minimum required driver versions:

- NVIDIA Turing architecture TU10x, TU11x - r440

- NVIDIA Ampere architecture GA100 - r450

- NVIDIA Ampere architecture GA100 MIG - r470 TRD1

- NVIDIA Ampere architecture GA10x - r455

Note

**Permissions:** Elevated permissions are required. On Linux use sudo to elevate privileges. On Windows the user must run from an admin command prompt or accept the UAC escalation dialog. See [Permissions Issues and Performance Counters](https://developer.nvidia.com/ERR_NVGPUCTRPERM) for more information.

Note

**Tensor Core:** If you run `nsys`` ``profile`` ``--gpu-metrics-devices`` ``all`, the Tensor Core utilization can be found in the GUI under the **SM instructions/Tensor Active** row.

Note that it is not practical to expect a CUDA kernel to reach 100% Tensor Core utilization since there are other overheads. In general, the more computation-intensive an operation is, the higher Tensor Core utilization rate the CUDA kernel can achieve.

#### Launching GPU Metrics from the CLI

GPU Metrics feature is controlled with 3 CLI switches:

- `--gpu-metrics-devices=[all,`` ``cuda-visible,`` ``none,`` ``<index>]` selects GPUs to sample (default is none).

- `--gpu-metrics-set=[<alias>,`` ``file:<file`` ``name>]` selects metric set to use (default is the 1st suitable from the list).

- `--gpu-metrics-frequency=[10..200000]` selects sampling frequency in Hz (default is 10000).

To profile with default options and sample GPU Metrics on GPU 1:

``` cpp
# Must have elevated permissions (see https://developer.nvidia.com/ERR_NVGPUCTRPERM) or be root (Linux) or Administrator (Windows)
$ nsys profile --gpu-metrics-devices=1 ./my-app
```

To list available GPUs, use:

``` cpp
$ nsys profile --gpu-metrics-devices=help
Possible --gpu-metrics-devices values are:
    1: Turing TU104 | GeForce RTX 2070 SUPER PCI[0000:65:00.0]
    all: Select all supported GPUs
    cuda-visible: Select GPUs that match CUDA_VISIBLE_DEVICES
    none: Disable GPU Metrics [Default]
Some GPUs are not supported:
    0: Volta GV100 | Quadro GV100 PCI[0000:17:00.0]
See the user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics
```

By default, the first **metric set** which supports all selected GPUs is used. You can manually select another metric set from the list. To see metrics sets available for the selected GPUs, use:

``` cpp
$ nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help
Possible --gpu-metrics-set values are:
    tu10x        : General Metrics for NVIDIA TU10x (any frequency)
    tu10x-gfxt   : Graphics Throughput Metrics for NVIDIA TU10x (frequency >= 10kHz)
    file:<file name> : use metric set from a given file
```

By default, **sampling frequency** is set to 10 kHz. But you can manually set it from 10 Hz to 200 kHz using the following:

``` cpp
--gpu-metrics-frequency=<value>
```

#### Launching GPU Metrics from the GUI

For commands to launch GPU Metrics from the CLI with examples, see [Profiling from the CLI](#cli-profiling).

When launching analysis in Nsight Systems, select **Collect GPU Metrics**.



Select the **GPUs** dropdown to pick which GPUs you wish to sample.

Select the **Metric set:** dropdown to choose which available metric set you would like to sample.



Note

Metric sets for GPUs that are not being sampled will be greyed out.

#### Sampling frequency

Sampling frequency can be selected from the range of 10 Hz - 200 kHz. The default value is 10 kHz.

The maximum sampling frequency without buffer overflow events depends on GPU (SM count), GPU load intensity, and overall system load. The bigger the chip and the higher the load, the lower the maximum frequency. If you need higher frequency, you can increase it until you get “Buffer overflow” message in the Diagnostics Summary report page.

Each metric set has a recommended sampling frequency range in its description. These ranges are approximate. If you observe `Inconsistent`` ``Data` or `Missing`` ``Data` ranges on timeline, please try closer to the recommended frequency.

#### Available metrics

- **GPC Clock Frequency** - `gpc__cycles_elapsed.avg.per_second`

  The average GPC clock frequency in hertz. In public documentation the GPC clock may be called the “Application” clock, “Graphic” clock, “Base” clock, or “Boost” clock.

  Note

  The collection mechanism for GPC can result in a small fluctuation between samples.

- **SYS Clock Frequency** - `sys__cycles_elapsed.avg.per_second`

  The average SYS clock frequency in hertz. The GPU front end (command processor), copy engines, and the performance monitor run at the SYS clock. On Turing and NVIDIA GA100 GPUs, the sampling frequency is based upon a period of SYS clocks (not time) so samples per second will vary with SYS clock. On NVIDIA GA10x GPUs, the sampling frequency is based upon a fixed frequency clock. The maximum frequency scales linearly with the SYS clock.

- **GR Active** - `gr__cycles_active.sum.pct_of_peak_sustained_elapsed`

  The percentage of cycles the graphics/compute engine is active. The graphics/compute engine is active if there is any work in the graphics pipe or if the compute pipe is processing work.

  GA100 MIG - MIG is not yet supported. This counter will report the activity of the primary GR engine.

- **Sync Compute In Flight** - `gr__dispatch_cycles_active_queue_sync.avg.pct_of_peak_sustained_elapsed`

  The percentage of cycles with synchronous compute in flight.

  CUDA: CUDA will only report synchronous queue in the case of MPS configured with 64 sub-context. Synchronous refers to work submitted in VEID=0.

  Graphics: This will be true if any compute work submitted from the direct queue is in flight.

- **Async Compute in Flight** - `gr__dispatch_cycles_active_queue_async.avg.pct_of_peak_sustained_elapsed`

  The percentage of cycles with asynchronous compute in flight.

  CUDA: CUDA will only report all compute work as asynchronous. The one exception is if MPS is configured and all 64 sub-context are in use. 1 sub-context (VEID=0) will report as synchronous.

  Graphics: This will be true if any compute work submitted from a compute queue is in flight.

- **Draw Started** - `fe__draw_count.avg.pct_of_peak_sustained_elapsed`

  The ratio of draw calls issued to the graphics pipe to the maximum sustained rate of the graphics pipe.

  Note

  The percentage will always be very low as the front end can issue draw calls significantly faster than the pipe can execute the draw call. The rendering of this row will be changed to help indicate when draw calls are being issued.

- **Dispatch Started** - `gr__dispatch_count.avg.pct_of_peak_sustained_elapsed`

  The ratio of compute grid launches (dispatches) to the compute pipe to the maximum sustained rate of the compute pipe.

  Note

  The percentage will always be very low as the front end can issue grid launches significantly faster than the pipe can execute the draw call. The rendering of this row will be changed to help indicate when grid launches are being issued.

- **Vertex/Tess/Geometry Warps in Flight** - `tpc__warps_active_shader_vtg_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of active vertex, geometry, tessellation, and meshlet shader warps resident on the SMs to the maximum number of warps per SM as a percentage.

- **Pixel Warps in Flight** - `tpc__warps_active_shader_ps_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of active pixel/fragment shader warps resident on the SMs to the maximum number of warps per SM as a percentage.

- **Compute Warps in Flight** - `tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of active compute shader warps resident on the SMs to the maximum number of warps per SM as a percentage.

- **Active SM Unused Warp Slots** - `tpc__warps_inactive_sm_active_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of inactive warp slots on the SMs to the maximum number of warps per SM as a percentage. This is an indication of how many more warps may fit on the SMs if occupancy is not limited by a resource such as max warps of a shader type, shared memory, registers per thread, or thread blocks per SM.

- **Idle SM Unused Warp Slots** - `tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of inactive warps slots due to idle SMs to the the maximum number of warps per SM as a percentage.

  This is an indicator that the current workload on the SM is not sufficient to put work on all SMs. This can be due to:

  - CPU starving the GPU.

  - Current work is too small to saturate the GPU.

  - Current work is trailing off but blocking next work.

- **SMs Active** - `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles SMs had at least 1 warp in flight (allocated on SM) to the number of cycles as a percentage. A value of 0 indicates all SMs were idle (no warps in flight). A value of 50% can indicate some gradient between all SMs active 50% of the sample period or 50% of SMs active 100% of the sample period.

- **SM Issue** - `sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles that SM sub-partitions (warp schedulers) issued an instruction to the number of cycles in the sample period as a percentage.

- **Tensor Active** - `sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the SM tensor pipes were active issuing tensor instructions to the number of cycles in the sample period as a percentage.

  TU102/4/6: This metric is not available on TU10x for periodic sampling. Please see Tensor Active/FP16 Active.

- **Tensor Active / FP16 Active** - `sm__pipe_shared_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed`

  TU102/4/6 only.

  The ratio of cycles the SM tensor pipes or FP16x2 pipes were active issuing tensor instructions to the number of cycles in the sample period as a percentage.

- **DRAM Read Bandwidth** - `dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__read_throughput.avg.pct_of_peak_sustained_elapsed`

- **VRAM Read Bandwidth** - `FBPA.TriageA.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageSCG.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageAC.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the DRAM interface was active reading data to the elapsed cycles in the same period as a percentage.

- **DRAM Write Bandwidth** - `dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__write_throughput.avg.pct_of_peak_sustained_elapsed`

- **VRAM Write Bandwidth** - `FBPA.TriageA.dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageSCG.dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageAC.dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the DRAM interface was active writing data to the elapsed cycles in the same period as a percentage.

- **NVENC Active**  
  `NVENC.TriageTop.nvenc__cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the NVENC unit was actively processing a command to the number of cycles in the same sample period as a percentage.

- **NVENC Read Throughput**  
  `NVENC.TriageTop.nvenc__memif2nvenc_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **NVENC Write Throughput**  
  `NVENC.TriageTop.nvenc__nvenc2memif_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the NVENC unit was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

- **OFA Active**  
  `OFA.TriageTop.ofa_cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the OFA (Optical Flow Accelerator) was actively processing a command to the number of cycles in the same sample period as a percentage.

- **OFA Read Throughput**  
  `OFA.TriageTop.ofa__memif2ofa_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **OFA Write Throughput**  
  `OFA.TriageTop.ofa__ofa2memif_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the OFA (Optical Flow Accelerator) was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

- **NVLink bytes received** - `nvlrx__bytes.avg.pct_of_peak_sustained_elapsed`

  The ratio of bytes received on the NVLink interface to the maximum number of bytes receivable in the sample period as a percentage. This value includes protocol overhead.

- **NVLink bytes transmitted** - `nvltx__bytes.avg.pct_of_peak_sustained_elapsed`

  The ratio of bytes transmitted on the NVLink interface to the maximum number of bytes transmittable in the sample period as a percentage. This value includes protocol overhead.

- **PCIe Read Throughput** - `pcie__read_bytes.avg.pct_of_peak_sustained_elapsed`

  The ratio of bytes received on the PCIe interface to the maximum number of bytes receivable in the sample period as a percentage. The theoretical value is calculated based upon the PCIe generation and number of lanes. This value includes protocol overhead.

- **PCIe Write Throughput** - `pcie__write_bytes.avg.pct_of_peak_sustained_elapsed`

  The ratio of bytes transmitted on the PCIe interface to the maximum number of bytes receivable in the sample period as a percentage. The theoretical value is calculated based upon the PCIe generation and number of lanes. This value includes protocol overhead.

- **PCIe Read Requests to BAR1** - `pcie__rx_requests_aperture_bar1_op_read.sum`

- **PCIe Write Requests to BAR1** - `pcie__rx_requests_aperture_bar1_op_write.sum`

  BAR1 is a PCI Express (PCIe) interface used to allow the CPU or other devices to directly access GPU memory. The GPU normally transfers memory with its copy engines, which would not show up as BAR1 activity. The GPU drivers on the CPU do a small amount of BAR1 accesses, but heavier traffic is typically coming from other technologies.

  On Linux, technologies like GPU Direct, GPU Direct RDMA, and GPU Direct Storage transfer data across PCIe BAR1. In the case of GPU Direct RDMA, that would be an Ethernet or InfiniBand adapter directly writing to GPU memory.

  On Windows, Direct3D12 resources can also be made accessible directly to the CPU via NVAPI functions to support small writes or reads from GPU buffers, in this case too many BAR1 accesses can indicate a performance issue, like it has been demonstrated in the Optimizing DX12 Resource Uploads to the GPU Using CPU-Visible VRAM technical blog post.

#### Exporting and Querying Data

It is possible to access metric values for automated processing using the Nsight Systems CLI export capabilities.

An example that extracts values of **SMs Active**:

``` bash
$ nsys export -t sqlite report.nsys-rep
$ sqlite3 report.sqlite "SELECT timestamp, value FROM GPU_METRICS
   JOIN TARGET_INFO_GPU_METRICS USING (metricId) WHERE value != 0
   AND metricName LIKE \"SMs Active%\" LIMIT 10;"

309277039|80
309301295|99
309325583|99
309349776|99
309373872|60
309397872|19
309421840|100
309446000|100
309470096|100
309494161|99
```

Values are integer percentages (0..100).

#### Limitations

- If metric sets with NVLink are used but the links are not active, they may appear as fully utilized.

- Only one tool that subscribes to these counters can be used at a time, therefore, Nsight Systems GPU Metrics feature cannot be used at the same time as the following tools:

  - Nsight Graphics

  - Nsight Compute

  - DCGM (Data Center GPU Manager)

    Use the following command:

    - `dcgmi`` ``profile`` ``--pause`

    - `dcgmi`` ``profile`` ``--resume`

    Or API:

    - `dcgmProfPause`

    - `dcgmProfResume`

  - Non-NVIDIA products which use:

    - CUPTI sampling used directly in the application. CUPTI trace is okay (although it will block Nsight Systems CUDA trace)

    - DCGM library

- Nsight Systems limits the amount of memory that can be used to store GPU Metrics samples. Analysis with higher sampling rates or on GPUs with more SMs has a risk of exceeding this limit. This will lead to gaps on timeline filled with `Missing`` ``Data` ranges. Future releases will reduce the frequency of this happening.

### NVML power and temperature metrics

Nsight Systems can now periodically sample power and temperature metrics from GPUs and plot them on the timeline in the GUI. These metrics are provided by the NVML API calls `nvmlDeviceGetPowerUsage` and `nvmlDeviceGetTemperature` respectively. The power metrics are provided in milliwatts (mW) and the temperature in degrees Celcius (C).

To enable the power and temperature sampling add the following option to the nsys `profile` or `start` commands:

``` cpp
--enable nvml_metrics[,arg1[=value1],arg2[=value2], ...]
```

There are no spaces following `nvml_metrics` plugin name. It is followed by a comma separated list of arguments or argument=value pairs. Arguments with spaces should be enclosed in double quotes.

Supported arguments are:

| Short name | Long name | Possible Parameters | Default | Switch Description |
|----|----|----|----|----|
| `-i` | `--interval` | integer | 100 | Sampling interval in milliseconds |
| `-h` | `--help` |  |  | Print help message |
| `-g` | `--gpu-devices` | all, cuda-visible, comma-separated GPU IDs list | cuda-visible | Set the GPUs to be sampled. cuda-visible will sample the GPUs set with CUDA_VISIBLE_DEVICES. An empty CUDA_VISIBLE_DEVICES will result in all GPUs being sampled. all and a GPU IDs list will precede CUDA_VISIBLE_DEVICES. |

**Usage Examples**

- `nsys`` ``profile`` ``--enable`` ``nvml_metrics`` ``...`  
  Sample power and temperature on all available GPUs every 100ms.

- `nsys`` ``profile`` ``--enable`` ``nvml_metrics,-i10`  
  Sample power and temperature on all available GPUs every 10ms.

For general information on Nsight Systems plugins please refer to [Nsight Systems Plugins](#nsight-systems-plugins) system.

### DCGM

NVIDIA Data Center GPU Manager (DCGM) is a suite of tools for managing and monitoring NVIDIA Datacenter GPUs in cluster environments. It includes active health monitoring, comprehensive diagnostics, system alerts, and governance policies including power and clock management. Infrastructure teams can use it standalone and in addition easily integrate it into cluster management tools, resource scheduling, and monitoring products from NVIDIA partners. For more information, see [DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/index.html)

Nsight Systems can now directly access metrics from DCGM and integrate them with other data collected, displaying on the timeline or making available for further analysis.

Supported arguments are:

| Option | Available Parameters (default in bold) | Switch Description |
|----|----|----|
| `--cpu` | all, help, dcgm:\<ID\> dcgm:\<ID1-ID2\> | Indicate which CPU(s) you are interested in or all. If you want to build a combination of CPUs, you can build a comma-separated combo like `dcgm:0,dcgm:2-4`. Help will give a list of available CPUs. |
| `--gpu` | all, cuda-visible, help, dcgm:\<ID\>, dcgm:\<ID1-ID2\> | Indicate which GPUs you are interested in, or all. If you want to build a combination of GPUs, you can build a comma-separated combo like `dcgm:0,dcgm:2-4`. Help will give a list of available GPUs. |
| `--hostengine-addr` | host:port | Address of the DCGM hostengine (host:port) or UDS path. When not specified, the plugin connects to localhost:5555 and falls back to embedded DCGM if that connection fails. |
| `--metrics` | comma-separated list of metric names or help | Give the list of DCGM metrics that you would like to collect, help will give the list of available metrics for your platform. If no metrics are selected, the tool will collect summary data for the CPUs, GPUs, and nvswitches on the system. See default metric options below. |
| `--nic` | all, help, dcgm:\<ID\> dcgm:\<ID1-ID2\> | Indicate which NIC(s) you are interested in or all. If you want to build a combination of NICs, you can build a comma-separated combo like `dcgm:0,dcgm:2-4`. Help will give a list of available NICs. |
| `--nvswitch` | all, help, dcgm:\<ID\> dcgm:\<ID1-ID2\> | Indicate which nvswitch(es) you are interested in or all. If you want to build a combination of nvswitch(es), you can build a comma-separated combo like `dcgm:0,dcgm:2-4`. Help will give a list of available nvswitch(es). |
| `--sampling-period` | milliseconds | Sampling period in milliseconds. Minimum allowed value is 100ms. |

Note

Unlike in most Nsight Systems collectors, the output from this command will not be sent to the standard CLI output, but rather to the output location for DCGM.

**Example 1 - Specifying Metrics**

``` text
nsys profile --enable=dcgm,--metrics=gpu_temp,power_usage_instant,sm_active,
   cpu_power_utilization,cpu_temp,--cpu=dcgm:0,--gpu=dcgm:0,--sampling-period=100
   appname
```

Screenshot:



**Example 2 - Using Presets**

The `s-` prefix selects preset metric groups defined in counter_groups.yaml; available presets are `gpu`, `gpu-perf`, `cpu`, `nvswitch`, and `power_smoothing` (for example, `s-gpu-perf` or `s-nvswitch`).

``` text
nsys profile --enable=dcgm,--metrics=s-gpu-perf,s-nvswitch
   ./nccl-tests/build/alltoall_perf -g 8 -n 100 -b 1M -e 32G -f 2
```

Screenshot:



For general information on Nsight Systems plugins please refer to [Nsight Systems Plugins](#nsight-systems-plugins).

## SoC Metrics

### Overview

SoC Metrics feature is intended to identify performance limiters in applications running on NVIDIA SoCs and is similar to GPU Metrics.

Nsight Systems SoC Metrics is only available for Linux and QNX targets on aarch64. It requires NVIDIA Orin architecture or newer.

### Available metrics

- **CPU Read Throughput**  
  `mcc__dram_throughput_srcnode_cpu_op_read.avg.pct_of_peak_sustained_elapsed`

  **CPU Write Throughput**  
  `mcc__dram_throughput_srcnode_cpu_op_write.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the SoC memory controllers were actively processing read/write operations from the CPU to the number of cycles in the same sample period as a percentage.

- **GPU Read Throughput**  
  `mcc__dram_throughput_srcnode_gpu_op_read.avg.pct_of_peak_sustained_elapsed`

  **GPU Write Throughput**  
  `mcc__dram_throughput_srcnode_gpu_op_write.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the SoC memory controllers were actively processing read/write operations from the GPU to the number of cycles in the same sample period as a percentage.

- **DBB Read Throughput**  
  `mcc__dram_throughput_srcnode_dbb_op_read.avg.pct_of_peak_sustained_elapsed`

  **DBB Write Throughput**  
  `mcc__dram_throughput_srcnode_dbb_op_write.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the SoC memory controllers were actively processing read/write operations from not-CPU/not-GPU to the number of cycles in the same sample period as a percentage.

- **DRAM Read Throughput**  
  `mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed`

  **DRAM Write Throughput**  
  `mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the SoC memory controllers were actively processing read/write operations to the number of cycles in the same sample period as a percentage.

- **DLA0/DLA1 Active**  
  `nvdla__cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the DLA (Deep Learning Accelerator) was actively processing a command to the number of cycles in the same sample period as a percentage.

- **DLA0/DLA1 Read Throughput**  
  `nvdla__dbb2nvdla_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **DLA0/DLA1 Write Throughput**  
  `nvdla__nvdla2dbb_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the DLA (Deep Learning Accelerator) was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

- **NVENC Active**  
  `nvenc__cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the NVENC unit was actively processing a command to the number of cycles in the same sample period as a percentage.

- **NVENC Read Throughput**  
  `nvenc__memif2nvenc_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **NVENC Write Throughput**  
  `nvenc__nvenc2memif_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the NVENC unit was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

- **PVA VPU Active**  
  `pvavpu__vpu_cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the PVA (Programmable Vision Accelerator) VPU (Vector Processing Unit) was actively processing a command to the number of cycles in the same sample period as a percentage.

- **PVA DMA Read Throughput**  
  `pva__dbb2pvadma_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **PVA DMA Write Throughput**  
  `pva__pvadma2dbb_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the PVA (Programmable Vision Accelerator) VPU (Vector Processing Unit) was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

  > Note
  >
  > To enable PVA trace on DRIVE 6.0.8.0, run these two commands before mounting any additional partitions:
  >
  > `echo`` ``1`` ``>/dev/nvpvadebugfs/pva0/tracing` `echo`` ``2`` ``>/dev/nvpvadebugfs/pva0/trace_level`

- **OFA Active**  
  `ofa_cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the OFA (Optical Flow Accelerator) was actively processing a command to the number of cycles in the same sample period as a percentage.

- **OFA Read Throughput**  
  `ofa__memif2ofa_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **OFA Write Throughput**  
  `ofa__ofa2memif_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the OFA (Optical Flow Accelerator) was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

- **VIC Active**  
  `vic_cycles_active.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the VIC (Video Image Compositor) was actively processing a command to the number of cycles in the same sample period as a percentage.

- **VIC Read Throughput**  
  `vic__dbb2vic_read_throughput.avg.pct_of_peak_sustained_elapsed`

  **VIC Write Throughput**  
  `vic__vic2dbb_write_throughput.avg.pct_of_peak_sustained_elapsed`

  The ratio of cycles the VIC (Video Image Compositor) was actively processing read/write operations to the number of cycles in the same sample period as a percentage.

### Launching SoC Metrics from the CLI

SoC Metrics feature is controlled with 3 CLI switches:

- `--soc-metrics=[true,`` ``false]` enables SoC Metrics sampling (default is false)

- `--soc-metrics-set=[<alias>,`` ``file:<file`` ``name>]` selects metric set to use (default is the 1st suitable from the list)

- `--soc-metrics-frequency=[100..200000]` selects sampling frequency in Hz (default is 10000)

To profile with default options:

``` cpp
# Must be root or added to 'debug' group
$ nsys profile --soc-metrics=true ./my-app
```

### Launching SoC Metrics from the GUI

When launching analysis in Nsight Systems, select **Collect SoC Metrics**.

The settings are similar to [GPU Metrics](index.html#launching-gpu-metrics-from-the-gui).

For commands to launch SoC Metrics from the CLI with examples, see the [CLI documentation](index.html#profiling-from-the-cli).

## CPU Profiling on Linux

Nsight Systems on Linux targets, utilizes the Linux OS’ perf subsystem to sample CPU Instruction Pointers (IPs) and backtraces, trace CPU context switches, and sample CPU and OS event counts. The Linux perf tool utilizes the same perf subsystem.

Nsight Systems Embedded Platforms Edition on Linux kernel prior to v5.15 uses a custom kernel module to collect the same data. The Nsight Systems CLI command `nsys`` ``status`` ``--environment` indicates when the kernel module is used instead of the Linux OS’ perf subsystem.

### Features

- **CPU Instruction Pointer / Backtrace Sampling**

  Nsight Systems can sample CPU Instruction Pointers / backtraces periodically. The collection of a sample is triggered by a hardware event overflow - e.g., a sample is collected after every 1 million CPU reference cycles on a per thread basis. In the GUI, samples are shown on the individual thread timelines, in the Event Viewer, and in the Top Down, Bottom Up, or Flat views which provide histogram-like summaries of the data. IP / backtrace collections can be configured in process-tree or system-wide mode. In process-tree mode, Nsight Systems will sample the process, and any of its descendants, launched by the tool. In system-wide mode, Nsight Systems will sample all processes running on the system, including any processes launched by the tool.

- **CPU Context Switch Tracing**

  Nsight Systems can trace every time the OS schedules a thread on a logical CPU and every time the OS thread gets unscheduled from a logical CPU. The data is used to show CPU utilization and OS thread utilization within the Nsight Systems GUI. Context switch collections can be configured in process-tree or system-wide mode. In process-tree mode, Nsight Systems will trace the process, and any of its descendants, launched by Nsight Systems. In system-wide mode, Nsight Systems will trace all processes running on the system, including any processes launched by the Nsight Systems.

- **CPU Event Sampling**

  Nsight Systems can periodically sample CPU hardware event counts and OS event counts and show the event’s rate over time in the Nsight Systems GUI. Event sample collections can be configured in system-wide mode only. In system-wide mode, Nsight Systems will sample event counts of all CPUs and the OS event counts running on the system. Event counts are not directly associated with processes or threads.

- **CPU Core Metrics**

  Nsight Systems can access and make available information about CPU core metrics. The `--cpu-metrics=help` command will list available metrics. Then selected options can be fed into the `--cpu-metrics` switch by name or by alias. These metrics can be used to determine how the CPU is oversubscribed. For example, see the [Grace Performance Tuning Guide](https://docs.nvidia.com/grace-performance-tuning-guide.pdf).

  In this version of Nsight Systems `--cpu-metrics` is available only on Linux and only for NVIDIA Grace CPU, NVIDIA GB10 Grace Blackwell Superchip (for example, on NVIDIA DGX Spark), and NVIDIA Thor (for example, in NVIDIA Jetson AGX Thor).

### System Requirements

- **Paranoid Level**

  The [system’s paranoid level](https://www.kernel.org/doc/Documentation/sysctl/kernel.txt) must be 2 or lower.

| Paranoid Level | CPU IP/backtrace Sampling process-tree mode | CPU IP/backtrace Sampling system-wide mode | CPU Context Switch Tracing process-tree mode | CPU Context Switch Tracing system-wide mode | Event Sampling system-wide mode |
|----|----|----|----|----|----|
| 3 or greater | not available | not available | not available | not available | not available |
| 2 | User mode IP/backtrace samples only | not available | available | not available | not available |
| 1 | Kernel and user mode IP/backtrace samples | not available | available | not available | not available |
| 0, -1 | Kernel and user mode IP/backtrace samples | Kernel and user mode IP/backtrace samples | available | available | hardware and OS events |

- **Kernel Version**

  To support the CPU profiling features utilized by Nsight Systems, the kernel version must be greater than or equal to v4.3. RedHat has backported the required features to the v3.10.0-693 kernel. RedHat distros and their derivatives (e.g. CentOS) require a 3.10.0-693 or later kernel. Use the `uname`` ``-r` command to check the kernel’s version.

- **perf_event_open syscall**

  The perf_event_open syscall needs to be available. When running within a Docker container, the default seccomp settings will normally block the perf_event_open syscall. To workaround this issue, use the Docker `run`` ``--privileged` switch when launching the docker or modify the docker’s seccomp settings. Some VMs (virtual machines), e.g. AWS, may also block the perf_event_open syscall.

- **Sampling Trigger**

  In some rare case, a sampling trigger is not available. The sampling trigger is either a hardware or software event that causes a sample to be collected. Some VMs block hardware events from being accessed and therefore, prevent hardware events from being used as sampling triggers. In those cases, Nsight Systems will fall back to using a software trigger if possible.

- **Checking Your Target System**

  Use the `nsys`` ``status`` ``--environment` command to check if a system meets the Nsight Systems CPU profiling requirements. Example output from this command is shown below. Note that this command does not check for Linux capability overrides - i.e., if the user or executable files have `CAP_SYS_ADMIN` or `CAP_PERFMON` capability. Also, note that this command does not indicate if system-wide mode can be used.

  

### Configuring a CPU Profiling Collection

When configuring Nsight Systems for CPU Profiling from the CLI, use some or all of the following options: `--sample`, `--cpuctxsw`, `--event-sample`, `--backtrace`, `--cpu-core-events`, `--event-sampling-interval`, `--os-events`, `--samples-per-backtrace`, and `--sampling-period`.

Details about these options, including examples can be found at [Profiling from the CLI](#cli-profiling).

When configuring from the GUI, the following options are available:



The configuration used during CPU profiling is documented in the Analysis Summary:



As well as in the Diagnosics Summary:



### Visualizing CPU Profiling Results

Here are example screenshots visualizing CPU profiling results. For details about navigating the Timeline View and the backtraces, see the section on [Timeline View in the Reading Your Report in the GUI section of the User Guide](index.html#timeline-view).

Example of CPU IP/Backtrace Data



In the timeline, yellow-orange marks can be found under each thread’s timeline that indicate the moment an IP / backtrace sample was collected on that thread (e.g., see the yellow-orange marks in the Specific Samples box above). Hovering the cursor over a mark will cause a tooltip to display the backtrace for that sample.

Below the Timeline is a drop-down list with multiple options including Events View, Top-Down View, Bottom-Up View, and Flat View. All four of these views can be used to view CPU IP / back trace sampling data.

Example of Event Sampling



Event sampling samples hardware or software event counts during a collection and then graphs those events as rates on the Timeline. The above screenshot shows four hardware events. Core and cache events are graphed under the associated CPU row (see the red box in the screenshot) while uncore and OS events are graphed in their own row (see the green box in the screenshot). Hovering the cursor over an event sampling row in the timeline shows the event’s rate at that moment.

### CPU Event Counters/Core Metrics

Nsight Systems can access and make available information about activities on the CPU. What exact data is available varies by the CPU and by the architecture.

| nsys profile/start command | Grace CPU | Future NVIDIA CPUs | Intel CPUs | AMD CPUs | Non-NVIDIA Arm-based CPUs |
|----|----|----|----|----|----|
| `--cpu-core-events` | full | n/a | extended | basic | basic |
| `--cpu-core-metrics` | full | n/a | n/a | n/a | n/a |
| `--cpu-metrics` (new) | full | full | extended | basic | basic |

Events Support:

- basic - only standard perf_event_open architecture-independent CPU core events

- extended - basic + a few common architecture-specific CPU core events

- full - basic + some architecture-specific CPU core events

Metrics Support:

- full - metrics derived from architecture-specific CPU core events

The `--cpu-metrics=help` command will list available events and metrics for your CPU. Then selected options can be fed into the `--cpu-metrics` switch by name or by alias.

Future versions of Nsight Systems will provide full event and metric support for other x86 and Arm architectures.

#### Event Multiplexing

There are hardware limitations on how many CPU counters can be collected at one time. If you need to collect more counters than are available, you can either perform multiple runs (as in the Arm Topdown methodology below) or you can take advantage of Nsight Systems’s support for event multiplexing.

To multiplex events, you need to define event groups. An event group can be defined in any of the following switches: `--os-events`, `--cpu-core-events`, `--cpu-core-metrics`, `--cpu-socket-events`, or `--cpu-socket-metrics`.

One new switch is added to support this feature but is not required. It is `--event-sampling-multiplex-interval`. If this switch is not set, the default interval is 2000 ms (i.e. 2 seconds). This switch defines when event group scheduling changes are made. The minimum `--event-sampling-multiplex-interval` is 250ms.

An event group is a group of events to be sampled concurrently. Use the ‘%’ delimiter to define an event group. For example;

``` text
--cpu-core-events=1,2,3%4,5,6
```

In this case, Nsight Systems will sample events 1,2, and 3 for 2 seconds, then switch the events 4,5, and 6 for 2 seconds, then switch back to events 1,2, and 3, etc. assuming the `--event-sampling-multiplex-interval` switch was not set.

``` text
nsys profile --os-events %3,2%% --cpu-socket-metrics 6,11%%14,15%
--cpu-core-events %%67,68%64,65,85,77,81 --event-sample system-wide
--event-sampling-interval 100 --event-sampling-multiplex-interval 500
--cpuctxsw system-wide -s none -t none -o five ../../ClockBenchmark
```

In this case, the following event groups were defined by the command line switches. The event groups were switched every 500ms. So, after 2 seconds, all of the events have been sampled and nsys reschedules group 0 to be sampled again.

- group 0 - events used by socket-metrics 6 and 11

- group 1 - os events 2 and 3

- group 2 - events used by socket-metrics 14 and 15, core events 67 and 68

- group 3 - core events 64,65,85,77, and 81

There is no limit to the number of event groups that can be defined. If an event group is empty, the command line will return an error. The events defined for an event group must fit in the available hardware PMUs as documented by the individual event switch help output. There is no limit on the number of concurrent os events.

> 

#### Arm Topdown Analysis

Arm Topdown methodology supports performance analysis, workload characterization, and microarchitecture exploration. You can find details on the technique at [Arm Topdown Methodology](https://developer.arm.com/documentation/109542/0100/Arm-Topdown-methodology).

Nsight Systems provides scripting to support running this analysis for the Grace (TM) and DGX Spark (TM) systems.

In your target-linux-sbsa-armv8/CpuProfiling directory, look for a script named `collect_cpu_topdown.sh`. This script simplifies collecting all PMU core event and metric data needed to perform a traditional CPU Topdown analysis of the workload’s CPU performance.

The script runs multiple system-wide `nsys`` ``profile` commands sequentially to collect the data. You can add additional Nsight Systems options to the command line as per usual, with the following exceptions:

- `--event-sample`, `--event-sampling-interval`, `--cpu-core-events`, and `--cpu-core-metrics` switches are set by the script for Topdown analysis.

- `-f`, `--force-overwrite` switch is set to true by the script

- `-o`, `--output` switch is set by the script to generate a list of predefined output nsys-rep files.

- `--kill` switch is set to the default value of `sigterm`

If an application is to be launched by the script, place a `--` between the nsys switches and the application command line.

Example command line:

``` text
collect_cpu_topdown.sh --trace=osrt,nvtx,cuda -- myApp arg1 arg2
```

Output files will be written to the current working directory. The output consists of a collection of .nsys-rep files that contain the metric data required to do a Topdown analysis of the workload. These files can be opened in the Nsight Systems GUI to view the metric results on the timeline.

You can futher use the NVTX CPU Topdown recipe (`nsys`` ``recipe`` ``nvtx_cpu_topdown`` ``--input`` ``.`) that will process the data from the .nsys-rep files and generate an output with CPU Topdown Methodology metrics computed for NVTX ranges. For details and use cases of this recipe, see [nvtx_cpu_topdown Recipe](../AnalysisGuide/index.html#nvtx-cpu-topdown-recipe).

Note

Arm Topdown analysis requires multiple system-wide collections and may take a significantly long time to run and post-process.

### Common Issues

- **Reducing Overhead Caused By Sampling**

  There are several ways to reduce overhead caused by sampling.

  - Disable sampling (i.e., use the `--sampling=none` switch).

  - Increase the sampling period (i.e., reduce the sampling rate) using the `--sampling-period` switch.

  - Stop collecting backtraces (i.e., use the `--backtrace=none` switch) or collect more efficient backtraces - if available, use the `--backtrace=lbr` switch.

  - reduce the number of backtraces collected per sample. See documentation for the `--samples-per-backtrace` switch.

- **Throttling**

  The Linux operating system enforces a maximum time to handle sampling interrupts. This means that if collecting samples takes more than a specified amount of time, the OS will throttle (i.e., slow down) the sampling rate to prevent the perf subsystem from causing too much overhead. When this occurs, sampling data may become irregular even though the thread is very busy.

  

  The above screenshot shows a case where CPU IP / backtrace sampling was throttled during a collection. Note the irregular intervals of sampling tickmarks on the thread timeline. The number of times a collection throttled is provided in the Nsight Systems GUI’s Diagnostics messages. If a collection throttles frequently (e.g., 1000s of times), increasing the sampling period should help reduce throttling.

  Note

  When throttling occurs, the OS sets a new (lower) maximum sampling rate in the procfs. This value must be reset before the sampling rate can be increased again. Use the following command to reset the OS’ max sampling rate `echo`` ``'100000'`` ``|`` ``sudo`` ``tee`` ``/proc/sys/kernel/perf_event_max_sample_rate`

- **Sample intervals are irregular**

  My samples are not periodic - why? My samples are clumped up - why? There are gaps in between the samples - why? Likely reasons:

  - Throttling, as described above.

  - The paranoid level is set to 2. If the paranoid level is set to 2, anytime the workload makes a system call and spends time executing kernel mode code, samples will not be collected and there will be gaps in the sampling data.

  - The sampling trigger itself is not periodic. If the trigger event is not periodic, for example, the Instructions Retired. event, sample collection will primarily occur when cache misses are occurring.

- **No CPU profiling data is collected**

  There are a few common issues that cause CPU profiling data to not be collected:

  - System requirements are not met. Check your system settings with the `nsys`` ``status`` ``--environment` command and see the System Requirements section above.

  - I profiled my workload in a Docker container but no sampling data was collected. By default, Docker containers prevent the perf_event_open syscall from being utilized. To override this behavior, launch the Docker with the `--privileged` switch or modify the Docker’s `seccomp` settings.

  - I profiled my workload in a Docker container running Ubuntu 20+ running on top of a host system running CentOS with a kernel version \< 3.10.0-693. The `nsys`` ``status`` ``--environment` command indicated that CPU profiling was supported. The host OS kernel version determines if CPU profiling is allowed and a CentOS host with a version \< 3.10.0-693 is too old. In this case, the `nsys`` ``status`` ``--environment` command is incorrect.

## NVIDIA Video Profiling

### NVIDIA Video Hardware Profiling

#### Limitations/Requirements

NVIDIA Video Hardware profiling requires:

- Linux (x86_64 or Arm) and Windows (x86_64)

- Only covers desktop platforms

- Driver version \>= 535

- GPU architecture Turing+

No NVIDIA Video Hardware profiling for:

- Mobile platforms

- Driver version \< 535

- GPU architecture \< Turing

- GSP is enabled and Driver \< 545.31

- MIG is enabled

- Confidential computing is enabled

- vGPU software \< 18.0

To learn more about GSP and on which GPUs it’s enabled by default, see the following [link](https://download.nvidia.com/XFree86/Linux-x86_64/550.142/README/gsp.html).

To turn off GSP permanently:

``` text
sudo su -c 'echo options nvidia NVreg_EnableGpuFirmware=0 > /etc/modprobe.d/nvidia-gsp.conf'
sudo update-initramfs -u # for Ubuntu-based systems
```

Then reboot.

Alternatively if you do not wish to reboot, this will disable until the next reboot:

``` text
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && \
sudo insmod /lib/modules/$(uname -r)/updates/dkms/nvidia.ko NVreg_EnableGpuFirmware=0
for i in $(seq 0 7); do sudo nvidia-smi -i $i -pm ENABLED; done
```

#### Running from the CLI

The feature is enabled through the `--gpu-video-devices` option. It is available from the `nsys`` ``profile`, `nsys`` ``launch` and `nsys`` ``start` commands.

The option behaves exactly like `--gpu-metrics-device` and accepts the following arguments:

- `--gpu-video-devices`` ``help` - List supported devices and their IDs, List unsupported devices (if any) and the reason.

- `--gpu-video-devices`` ``none` - Turn the feature off.

- `--gpu-video-devices`` ``all` - Enable the feature on all supported devices. An error is returned if no devices support the feature.

- `--gpu-video-devices`` ``<id1,id2,...>` - Enable the feature on the specified devices. The ID corresponds to what `help` returns. An error is returned if the ID is invalid.

Example:

``` text
$ nsys profile --gpu-video-devices help
Possible --gpu-video-devices values are:
    0: NVIDIA GeForce RTX 3070 PCI[0000:65:00.0]
    all: Select all supported GPUs
    none: Disable GPU video accelerator tracing [Default]

Some GPUs don't support video accelerator tracing:
    Quadro P620 PCI[0000:04:00.0] (reason = Arch Pascal < Turing)

See the user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
```

Note that this is a system-wide feature; i.e., it doesn’t require a program to be launched.

### NVIDIA Video Codec SDK Trace

Nsight Systems for x86 Linux and Windows targets can trace calls from the NVIDIA Video Codec SDK. This software trace can be launched from the GUI or using the `--trace`` ``nvvideo` from the CLI



On the timeline, calls on the CPU to the NVIDIA Encoder API, NVIDIA Decoder API, and NVIDIA nvJPEG API will be shown.



#### NV Encoder API Functions Traced by Default

``` cpp
NvEncodeAPICreateInstance
nvEncOpenEncodeSession
nvEncGetEncodeGUIDCount
nvEncGetEncodeGUIDs
nvEncGetEncodeProfileGUIDCount
nvEncGetEncodeProfileGUIDs
nvEncGetInputFormatCount
nvEncGetInputFormats
nvEncGetEncodeCaps
nvEncGetEncodePresetCount
nvEncGetEncodePresetGUIDs
nvEncGetEncodePresetConfig
nvEncGetEncodePresetConfigEx
nvEncInitializeEncoder
nvEncCreateInputBuffer
nvEncDestroyInputBuffer
nvEncCreateBitstreamBuffer
nvEncDestroyBitstreamBuffer
nvEncEncodePicture
nvEncLockBitstream
nvEncUnlockBitstream
nvEncLockInputBuffer
nvEncUnlockInputBuffer
nvEncGetEncodeStats
nvEndGetSequenceParams
nvEncRegisterAsyncEvent
nvEncUnregisterAsyncEvent
nvEncMapInputResource
nvEncUnmapInputResource
nvEncDestroyEncoder
nvEncInvalidateRefFrames
nvEncOpenEncodeSessionEx
nvEncRegisterResource
nvEncUnregisterResource
nvEncReconfigureEncoder
nvEncCreateMVBuffer
nvEncDestroyMVBuffer
nvEncRunMotionEstimationOnly
nvEncGetLastErrorString
nvEncSetIOCudaStreams
nvEncGetSequenceParamEx
```

#### NV Decoder API Functions Traced by Default

``` cpp
cuvidCreateVideoSource
cuvidCreateVideoSourceW
cuvidDestroyVideoSource
cuvidSetVideoSourceState
cudaVideoState
cuvidGetSourceVideoFormat
cuvidGetSourceAudioFormat
cuvidCreateVideoParser
cuvidParseVideoData
cuvidDestroyVideoParser
cuvidCreateDecoder
cuvidDestroyDecoder
cuvidDecodePicture
cuvidGetDecodeStatus
cuvidReconfigureDecoder
cuvidMapVideoFrame
cuvidUnmapVideoFrame
cuvidMapVideoFrame64
cuvidUnmapVideoFrame64
cuvidCtxLockCreate
cuvidCtxLockDestroy
cuvidCtxLock
cuvidCtxUnlock
```

#### NV JPEG API Functions Traced by Default

``` cpp
nvjpegBufferDeviceCreate
nvjpegBufferDeviceRetrieve
nvjpegBufferPinnedCreate
nvjpegBufferPinnedRetrieve
nvjpegCreate
nvjpegCreateEx
nvjpegCreateSimple
nvjpegDecode
nvjpegDecodeBatched
nvjpegDecodeBatchedEx
nvjpegDecodeBatchedInitialize
nvjpegDecodeBatchedPreAllocate
nvjpegDecodeBatchedSupported
nvjpegDecodeBatchedSupportedEx
nvjpegDecodeJpeg
nvjpegDecodeJpegDevice
nvjpegDecodeJpegHost
nvjpegDecodeJpegTransferToDevice
nvjpegDecodeParamsCreate
nvjpegDecodeParamsDestroy
nvjpegDecodeParamsSetAllowCMYK
nvjpegDecodeParamsSetOutputFormat
nvjpegDecodeParamsSetROI
nvjpegDecodeParamsSetScaleFactor
nvjpegDecoderCreate
nvjpegDecoderDestroy
nvjpegDecoderJpegSupported
nvjpegDecoderStateCreate
nvjpegDestroy
nvjpegEncodeGetBufferSize
nvjpegEncodeImage
nvjpegEncodeRetrieveBitstream
nvjpegEncodeRetrieveBitstreamDevice
nvjpegEncoderParamsCopyHuffmanTables
nvjpegEncoderParamsCopyMetadata
nvjpegEncoderParamsCopyQuantizationTables
nvjpegEncoderParamsCreate
nvjpegEncoderParamsDestroy
nvjpegEncoderParamsSetEncoding
nvjpegEncoderParamsSetOptimizedHuffman
nvjpegEncoderParamsSetQuality
nvjpegEncoderParamsSetSamplingFactors
nvjpegEncoderStateCreate
nvjpegEncoderStateDestroy
nvjpegEncodeYUV,(nvjpegHandle_t handle
nvjpegGetCudartProperty
nvjpegGetDeviceMemoryPadding
nvjpegGetImageInfo
nvjpegGetPinnedMemoryPadding
nvjpegGetProperty
nvjpegJpegStateCreate
nvjpegJpegStateDestroy
nvjpegJpegStreamCreate
nvjpegJpegStreamDestroy
nvjpegJpegStreamGetChromaSubsampling
nvjpegJpegStreamGetComponentDimensions
nvjpegJpegStreamGetComponentsNum
nvjpegJpegStreamGetFrameDimensions
nvjpegJpegStreamGetJpegEncoding
nvjpegJpegStreamParse
nvjpegJpegStreamParseHeader
nvjpegSetDeviceMemoryPadding
nvjpegSetPinnedMemoryPadding
nvjpegStateAttachDeviceBuffer
nvjpegStateAttachPinnedBuffer
```

## Network Communication Profiling

Nsight Systems can be used to profiles several popular network communication protocols and many network hardware components. To enable this, please select the **Network profiling options** dropdown.

Note

Network hardware profiling uses statistical sampling of counters on the various appliances. Network communication API profiling uses direct trace of relevant function calls. Nsight Systems correlates the data as well as we can, however the inherent profiling differences make correlation somewhat inexact.



Then select the libraries you would like to trace:



The corresponding Nsight Systems CLI `--trace|-t` options are `mpi`, `oshmem`, `ucx`, and `nccl`. For multi-node runs, please refer to the section on [Handling Application Launchers (mpirun, deepspeed, etc)](#handling-application-launchers).

### MPI API Trace

Nsight Systems has built-in API trace support for Open MPI and MPICH based MPI implementations via `--trace=mpi` or by selecting the *MPI* checkbox under *Network profiling options*. If the auto-detection of the MPI implementation fails, it is possible to specify it via `--mpi-impl=[openmpi|mpich]` or the respective checkbox in the GUI.

Nsight Systems will trace a subset of the MPI API, including blocking and non-blocking point-to-point and collective communications as well as MPI one-sided communications, file I/O, and pack operations (see [MPI functions traced](#mpi-functions-traced)).

If you require more control over the list of traced APIs or if you are using a different MPI implementation, you can use the [NVTX wrappers for MPI](https://github.com/NVIDIA/cuda-profiler/tree/master/nvtx_pmpi_wrappers) on GitHub. Choose an NVTX domain name other than “MPI,” since it is filtered out by Nsight Systems when MPI tracing is not enabled. Use the NVTX-instrumented MPI wrapper library as follows:

``` text
nsys profile -e LD_PRELOAD=${PATH_TO_YOUR_NVTX_MPI_LIB} --trace=nvtx
```



Note

If not all ranks are traced, `NSYS_MPI_STORE_TEAMS_PER_RANK` has to be set to `1`. If communicator tracking is still causing issues, it can be disabled by setting `NSYS_MPI_DISABLE_COMMUNICATOR_TRACKING=1`.

#### MPI Communication Parameters

Nsight Systems can get additional information about MPI communication parameters. Currently, the parameters are only visible in the mouseover tooltips or in the event log. This means that the data is only available via the GUI. Future versions of the tool will export this information into the SQLite data files for postrun analysis.

In order to fully interpret MPI communications, data for all ranks associated with a communication operation must be loaded into Nsight Systems.

Here is an example of `MPI_COMM_WORLD` data. This does not require any additional team data, since local rank is the same as global rank.

(Screenshot shows communication parameters for an MPI_Bcast call on rank 3.)



When not all processes that are involved in an MPI communication are loaded into Nsight Systems the following information is available.

- Right-hand screenshot shows a reused communicator handle (last number increased).

- Encoding: `MPI_COMM[\*team`` ``size\*]*global-group-root-rank\*.*group-ID\*`



When all reports are loaded into Nsight Systems:

- World rank is shown in addition to group-local rank “(world rank X).”

- Encoding: MPI_COMM\[\*team size\*\]{rank0, rank1, …}.

- At most 8 ranks are shown (the numbers represent world ranks, the position in the list is the group-local rank).



#### MPI functions traced

``` cpp
MPI_Init[_thread], MPI_Finalize
MPI_Send, MPI_{B,S,R}send, MPI_Recv, MPI_Mrecv
MPI_Sendrecv[_replace]

MPI_Barrier, MPI_Bcast
MPI_Scatter[v], MPI_Gather[v]
MPI_Allgather[v], MPI_Alltoall[{v,w}]
MPI_Allreduce, MPI_Reduce[_{scatter,scatter_block,local}]
MPI_Scan, MPI_Exscan

MPI_Isend, MPI_I{b,s,r}send, MPI_I[m]recv
MPI_{Send,Bsend,Ssend,Rsend,Recv}_init
MPI_Start[all]
MPI_Ibarrier, MPI_Ibcast
MPI_Iscatter[v], MPI_Igather[v]
MPI_Iallgather[v], MPI_Ialltoall[{v,w}]
MPI_Iallreduce, MPI_Ireduce[{scatter,scatter_block}]
MPI_I[ex]scan
MPI_Wait[{all,any,some}]

MPI_Put, MPI_Rput, MPI_Get, MPI_Rget
MPI_Accumulate, MPI_Raccumulate
MPI_Get_accumulate, MPI_Rget_accumulate
MPI_Fetch_and_op, MPI_Compare_and_swap

MPI_Win_allocate[_shared]
MPI_Win_create[_dynamic]
MPI_Win_{attach, detach}
MPI_Win_free
MPI_Win_fence
MPI_Win_{start, complete, post, wait}
MPI_Win_[un]lock[_all]
MPI_Win_flush[_local][_all]
MPI_Win_sync

MPI_File_{open,close,delete,sync}
MPI_File_{read,write}[_{all,all_begin,all_end}]
MPI_File_{read,write}_at[_{all,all_begin,all_end}]
MPI_File_{read,write}_shared
MPI_File_{read,write}_ordered[_{begin,end}]
MPI_File_i{read,write}[_{all,at,at_all,shared}]
MPI_File_set_{size,view,info}
MPI_File_get_{size,view,info,group,amode}
MPI_File_preallocate

MPI_Pack[_external]
MPI_Unpack[_external]
```

### OpenSHMEM Library Trace

If OpenSHMEM library trace is selected Nsight Systems will trace the subset of OpenSHMEM API functions that are most likely be involved in performance bottlenecks. To keep overhead low Nsight Systems does not trace all functions.

**OpenSHMEM 1.5 Functions Not Traced**

``` cpp
shmem_my_pe
shmem_n_pes
shmem_global_exit
shmem_pe_accessible
shmem_addr_accessible
shmem_ctx_{create,destroy,get_team}
shmem_global_exit
shmem_info_get_{version,name}
shmem_{my_pe,n_pes,pe_accessible,ptr}
shmem_query_thread
shmem_team_{create_ctx,destroy}
shmem_team_get_config
shmem_team_{my_pe,n_pes,translate_pe}
shmem_team_split_{2d,strided}
shmem_test*
```

### UCX API Trace

If UCX API trace is selected Nsight Systems will trace the subset of functions of the UCX protocol layer UCP that are most likely be involved in performance bottlenecks. To keep overhead low Nsight Systems does not trace all functions.

The following environment variables control what is recorded:

- `NSYS_UCP_COMM_SUBMIT`: (enabled by default) If set to `0`, UCP communication submission calls are not recorded any more. These calls are usually short, because the communication itself is handled in a worker thread.

- `NSYS_UCP_COMM_PROGRESS`: (enabled by default) If set to `0`, tracking of (process-local) UCP communication progress is disabled. The progress tracking uses UCP completion callbacks.

- `NSYS_UCP_COMM_PARAMS`: (enabled by default) If set to `0`, UCP communication parameters (tag, remote worker UID, packed message size, buffer address) will not be recorded. Recording the remote worker UID requires UCX \>= 1.12.0. Recording the packed message size requires UCX \>= 1.14.0.

#### UCX functions traced

``` cpp
ucp_am_send_nb[x]
ucp_am_recv_data_nbx
ucp_am_data_release
ucp_atomic_{add{32,64},cswap{32,64},fadd{32,64},swap{32,64}}
ucp_atomic_{post,fetch_nb,op_nbx}
ucp_cleanup
ucp_config_{modify,read,release}
ucp_disconnect_nb
ucp_dt_{create_generic,destroy}
ucp_ep_{create,destroy,modify_nb,close_nbx}
ucp_ep_flush[{_nb,_nbx}]
ucp_listener_{create,destroy,query,reject}
ucp_mem_{advise,map,unmap,query}
ucp_{put,get}[_nbi]
ucp_{put,get}_nb[x]
ucp_request_{alloc,cancel,is_completed}
ucp_rkey_{buffer_release,destroy,pack,ptr}
ucp_stream_data_release
ucp_stream_recv_data_nb
ucp_stream_{send,recv}_nb[x]
ucp_stream_worker_poll
ucp_tag_msg_recv_nb[x]
ucp_tag_{send,recv}_nbr
ucp_tag_{send,recv}_nb[x]
ucp_tag_send_sync_nb[x]
ucp_worker_{create,destroy,get_address,get_efd,arm,fence,wait,signal,wait_mem}
ucp_worker_flush[{_nb,_nbx}]
ucp_worker_set_am_{handler,recv_handler}
```

**UCX Functions Not Traced:**

``` cpp
ucp_config_print
ucp_conn_request_query
ucp_context_{query,print_info}
ucp_get_version[_string]
ucp_ep_{close_nb,print_info,query,rkey_unpack}
ucp_mem_print_info
ucp_request_{check_status,free,query,release,test}
ucp_stream_recv_request_test
ucp_tag_probe_nb
ucp_tag_recv_request_test
ucp_worker_{address_query,print_info,progress,query,release_address}
```

Additional API functions from other UCX layers may be added in a future version of the product.

### NVIDIA NCCL Trace

Nsight Systems provides two methods for tracing NVIDIA NCCL (NVIDIA Collective Communications Library) operations:

1.  **Legacy NCCL tracing:** is based on NVTX annotations within the NCCL library itself.

    - Enabled by default when NVTX tracing is active

    - Traces API calls on the CPU

    - Provides limited GPU-projection of ranges in the GUI

2.  **Advanced NCCL tracing:** A more detailed tracing mechanism introduced in Nsight Systems 2025.6.1.

    - Requires NCCL version 2.28 or higher (with limited support for versions 2.27.4 and later)

    - Support for Copy Engine (CE) collectives requires NCCL version 2.29 or higher

    - Provides detailed information about GPU operations and asynchronous runtime scheduling

    - Enhances correlation across events

    - Less precise timestamps for CPU API calls compared to legacy NCCL tracing

#### NCCL Execution Concepts

To effectively interpret NCCL traces, it is important to understand the following aspects of NCCL’s operation. A NCCL collective operation comprises multiple steps on the CPU and GPU:

- The application calls the NCCL API.

- The NCCL runtime prepares and schedules the operation in queues for the GPU.

- The CUDA kernel is launched.

- The operation executes within a CUDA kernel.

The order of these steps and the threads in which they occur depend on the application pattern:

- Use of groups

- Blocking vs. non-blocking communicators

- CUDA graph capture

**Group Operations**

When using NCCL groups, all operations within the group are executed at the end of the group:

- Operations in a group are typically fused into a single CUDA kernel per rank/device.

- With legacy NCCL tracing, the `ncclGroupEnd` function is projected to the fused CUDA kernel on the GPU.

- When no explicit groups are used, there is an implicit group as part of each API call. This implicit grouping is not shown in legacy NCCL tracing.

**Non-blocking Communicators**

For non-blocking communicators, CUDA calls are performed in different threads:

- Legacy NCCL tracing cannot track these cross-thread operations.

- Plugin-based tracing properly correlates all events (e.g., API calls, CUDA calls, and GPU operations) that belong to one logical operation.

**Graph Capture**

With graph capture, the API calls and kernel launch are captured once, but the runtime scheduling and GPU operations happen multiple times, once per graph launch.

#### Advanced NCCL Tracing

The advanced NCCL tracing mechanism provides comprehensive visibility into NCCL operations across the CPU and GPU. The actual trace structure depends heavily on the specific application patterns.

Internally, the advanced NCCL tracing is built as a profiler plugin of NCCL (not to be confused with an nsys plugin).

##### Trace Information

The advanced NCCL tracing provides the following information in reports:

**API Calls**

NCCL API calls (collective, point-to-point, and group operations) on the CPU are traced with the following characteristics:

- Groups are shown as an `API`` ``Group` range that spans all API calls within the group. Technically, this range does not include `ncclGroupStart`, but extends through the end of `ncclGroupEnd`/`GroupLaunch`.

- Individual API calls are shown below the `API`` ``Group` range.

- The `nccl` prefix is omitted from function names (e.g., `ncclAllReduce` appears as `AllReduce`).

- The `GroupLaunch` range corresponds to the `ncclGroupEnd` function call on the application thread invoking the NCCL group functions.

  - For blocking communication, this range encompasses the preparation for kernel launches. There can be multiple launches per group, usually one per rank/device, and the actual CUDA kernel launches are collected as `KernelLaunch` ranges.

**Runtime Scheduling**

Runtime scheduling events show where NCCL runs on the CPU and queues operations for the GPU:

- `GroupRuntime` ranges encompass the individual runtime scheduling for collective and point-to-point operations.

- Runtime scheduling can occur in different contexts:

  - Directly at the end of the `API`` ``Group` for blocking communicators

  - In a separate thread for non-blocking communicators

  - For graph launches, there are multiple runtime groups for a single API group, one per launch, and the group events occur in a host function on a special thread

**GPU Operations**

GPU operations are displayed with the following details:

- Individual operations within fused CUDA kernels are shown with accurate GPU timestamps.

- Operations are further split across multiple channels that execute concurrently.

- GPU operations are shown directly under the GPU in the timeline view.

**Copy Engine (CE) Collectives**

CE-based collectives are not shown as GPU operations in the NCCL device row. They use CUDA memory operations rather than CUDA kernels. By default, for each API call to a CE collective, a corresponding range will be shown within the `GroupLaunch` range. In addition, you can configure Nsight Systems to add two `CE`` ``Sync` ranges and one `CE`` ``Batch` range below each CE collective range (see [Configuration Options](#nccl-trace-options)).

**Proxy Activity**

NCCL uses a proxy thread to support CPU-orchestrated inter-node communication. Some activity in this proxy thread can be collected and shown in the timeline view. Proxy activity recording is experimental, may change in future releases, and is not enabled by default. This view is intended for expert-level analysis and can be difficult to interpret. Use it for deep dives into network bottlenecks and supplement it with general network counters. In the Nsight Systems GUI, proxy step visualization can be dense due to the volume of detail and is best suited to analysis of exported files.

Proxy operation ranges encapsulate the proxy activity for one peer, channel and direction. Each proxy operation is split into proxy steps to process individual chunks in a pipelined manner. Within each proxy step, state changes are recorded. Due to the amount of data collected for proxy steps, the data collection can have a significant performance impact.

Proxy counters summarize the proxy activity within one communicator. They display the number of proxy steps in any given state, separated into `ProxyStepSend` and `ProxyStepRecv`.

All NCCL proxy activity is performed by a dedicated CPU thread for each rank/communicator. This thread is named `NCCL`` ``Progress`` ``[$rank/$nRanks]:`` ``$commHash` to easily identify the context. Nsight Systems shows both the proxy operations and proxy counters as part of this thread.

**Communicators**

All NCCL events are organized in categories: one for each communicator. In the GUI, NCCL rows initially show events from all communicators but can be expanded to display events grouped by communicator. For complex applications, developers should assign names to communicators within the application by setting `commName` in `ncclConfig_t`. Creation of communicators is indicated by a `CommInit` marker in a dedicated initialization thread.

**Metadata**

All NCCL events carry metadata (payload information):

- Information about the communicator,

- Operation-related events carry metadata about the specific operation, e.g., data type, element count, etc.,

- Correlation identifiers.

**Event Correlation**

Events corresponding to individual operations are correlated by the `apiId`, which for collective operations includes the operations of all participating ranks. The correlation links the API call, collective runtime scheduling, and GPU operation, helping to track operations through the UI. Additional correlation identifiers are available but not currently used to avoid highlighting too many ranges simultaneously:

- `apiGroupId`: All API calls and corresponding kernel launches in a specific (thread-local) group

- `group`: Group identifier

**CUDA Graph Capture**

The advanced NCCL tracing can track individual operations through CUDA graph capture and graph launches, from the API call through runtime scheduling in host functions to the GPU operation.

**Memory Overhead**

To collect a consistent and correlated trace, the NCCL injection allocates data for each event using a growing pre-allocated buffer. The buffer is freed when the communicator is finalized.

**Capture Ranges**

Capture ranges affect data collection. Outside of capture ranges, no data is collected and the overhead is significantly reduced. In particular, the NCCL injection does not allocate data for each event outside capture ranges. If correlating NCCL events are only partially inside a capture range, the correlation may be lost. For example, if the API calls occur before a capture range and only the operations are inside, the operations will be included but without correlation identifiers.

##### Report View

The timeline view of a report with advanced NCCL tracing includes several types of `NCCL` rows:

- A `NCCL` row directly under each device shows the GPU operations.

- CPU threads that call API functions include a `NCCL` row showing the respective API events.

- Internal threads that execute non-blocking or graph host functions also include a `NCCL` row showing runtime events (not shown in the example).

- An additional global `NCCL` row at the bottom combines the CPU-side NCCL events from all processes/threads in the report.

- Below the stream rows on the device, `NCCL` rows show projections of the respective internal NCCL ranges which were responsible for launching the corresponding CUDA activity.

Each `NCCL` row can be expanded to show individual events for each communicator.

The screenshot shows two participating devices with a simple API group that includes two API calls (one for each rank). Since this is a blocking communicator, the group directly executes the NCCL internal functions `GroupLaunch` and `GroupRuntime`. A tooltip shows the metadata of a NCCL operation.



##### Exporting

The data from advanced NCCL tracing can be exported as SQLite database for further processing. See [SQLite Schema Reference](../AnalysisGuide/index.html#sqlite-schema) for more details.

NCCL events are available as NVTX ranges in the exported database. While legacy NCCL tracing also uses NVTX, the exported events differ. Add `--include-json`` ``true` to include the metadata as JSON in the exported tables. The specific schema of the exported tables may be subject to change in the future.

##### Limitations

The advanced NCCL tracing has the following limitations (as of NCCL v2.28):

- Communicator creation API functions are not collected directly. However, the creation is indicated in the initialization thread via `CommInit` markers.

- Single-rank communications, which are usually only buffer copies, are not shown at all. Their events are not captured.

- Symmetric memory based collectives (new low latency implementations of AllReduce, AllGather and ReduceScatter) are not fully supported. Only the API calls are collected.

- AllToAll, Gather, Scatter operations will be shown as their underlying grouped point-to-point operations rather than the high-level API call.

##### Usage

To use advanced NCCL tracing:

1.  Enable NCCL tracing using the `-t`` ``nccl` option with `nsys`` ``profile` or enable the “NCCL” section in the Network profiling options of the GUI:

    ``` text
    nsys profile -t nccl <application>
    ```

    Note

    By default, only the legacy NCCL tracing (API calls) is active.

2.  Normally, both NCCL and CUDA tracing are enabled.

    ``` text
    nsys profile -t nccl,cuda <application>
    ```

3.  When advanced NCCL tracing is enabled, the legacy NCCL tracing is automatically disabled.

##### Configuration Options

Configure advanced NCCL tracing using the `--nccl-trace` option with `nsys`` ``profile`. The following values are available and can be combined using commas (e.g., `--nccl-trace=api,rt`):

- `api-group`: The `API`` ``Group` range

- `api-coll`: Collective API calls (e.g., `AllReduce`)

- `api-p2p`: Point-to-point API calls (`Send` and `Recv`)

- `api`: All of the three API ranges above (`api-group`, `api-coll`, `api-p2p`)

- `group`: `GroupRuntime` ranges for operation scheduling on the CPU

- `coll`: Runtime scheduling of individual collective operations

- `p2p`: Runtime scheduling of individual point-to-point operations

- `kernel-launch`: Ranges around CUDA kernel launches

- `ce-coll`: Launch of memory operations for Copy Engine (CE) -based collectives

- `ce-sync`: Synchronization of CE collectives (within ce-coll)

- `ce-batch`: Batch operations of CE collectives (within ce-coll)

- `rt`: All runtime ranges (`group`, `coll`, `p2p`, `kernel-launch`)

- `gpu`: Individual operations on the GPU

- `proxy-op`: Proxy operation ranges (experimental)

- `proxy-step`: Proxy step ranges including state changes (experimental, high overhead)

- `proxy-counters`: Proxy counters (experimental)

- `default`: The default set of events (`api`, `group`, `gpu`, `ce-coll`)

- `all`: All possible events (except `proxy-op`, `proxy-step`)

The interaction between `--nccl-trace` and `--trace` works as follows:

- If `--nccl-trace` is explicitly set, it takes priority over any NCCL-related settings from `--trace`. Any value other than `none` implicitly includes `nccl` in `--trace`.

- If `--nccl-trace` is not set, the default behavior depends on `--trace`: if `nccl` is included in `--trace`, `default` is used for `--nccl-trace`; otherwise, `none` is used.

##### More Examples

Enable advanced NCCL tracing with all events as well as CUDA tracing:

``` text
nsys profile -t nccl,cuda --nccl-trace=all <application>
```

Disable advanced NCCL tracing to fall back to legacy NCCL tracing:

``` text
nsys profile -t cuda,nvtx <application>
```

Disable the NCCL NVTX domain, neither advanced NCCL tracing nor legacy NCCL tracing will be enabled:

``` text
nsys profile -t cuda,nvtx --nvtx-domain-exclude=NCCL <application>
```

Disable NVTX tracing completely, neither advanced NCCL tracing nor legacy NCCL tracing will be enabled:

``` text
nsys profile -t cuda <application>
```

### NVIDIA NVSHMEM Trace

The NVIDIA network communication library NVSHMEM has been instrumented using NVTX annotations. To enable tracing this library in Nsight Systems, turn on NVTX tracing in the GUI or CLI. To enable the NVTX instrumentation of the NVSHMEM library, make sure that the environment variable `NVSHMEM_NVTX` is set properly; e.g., `NVSHMEM_NVTX=common`.

## Network Hardware Profiling

Nsight Systems can be used to profile several popular network communication protocols and many network hardware components.

Note

Network hardware profiling uses statistical sampling of counters on the various appliances. Network communication API profiling uses direct trace of relevant function calls. Nsight Systems correlates the data as well as we can, however the inherent profiling differences make correlation somewhat inexact.

### InfiniBand Network Information

**Overview**

By default, Nsight Systems displays low-level identifiers like LIDs (Local Identifiers) and GUIDs (Globally Unique Identifiers). Instead, Nsight Systems can leverage InfiniBand network information to display the actual names of nodes and switches. This makes the Nsight Systems reports much more intuitive and easier to understand at a glance.

InfiniBand network information discovery is done using the ibdiagnet utility. Either:

> - Run ibdiagnet and store the generated network information files to be later used by Nsight Systems.
>
>   - This method is useful for large networks, where  
>     ibdiagnet’s network discovery time may be long, and for networks where only administrators have permissions to query the network information.
>
> - A user can ask Nsight Systems to run ibdiagnet to collect the network information during the profiling session.
>
>   - This method is useful for small networks.

**Limitations/Requirements**

The user needs to have permission to send MADs (management datagrams). To check if you have permission to send MADs, check if you can access the `/dev/infiniband/umad*` files. To give user permissions to send MADs on RedHat systems, follow the directions at [RedHat Solutions](https://access.redhat.com/solutions/5929621).

**Relevant Switches**

The following Nsight Systems command line switches enable collecting InfiniBand network information:

> - `ib-net-info-devices`  
>   This should be followed by a comma separated list of NIC names, from which ibdiagnet will run network discovery. The results of the network discovery will be automatically loaded into Nsight Systems.
>
> - `ib-net-info-files`  
>   This should be followed by a comma separated list of pre-generated ibdiagnet db_csv file paths, which Nsight Systems will read.
>
> - `ib-net-info-output`  
>   This should be followed by a path of a directory into which Nsight Systems will store the ibdiagnet network discovery data. These files will be used by the `ib-net-info-devices` command line switch. This command line switch can only be used together with the `ib-net-info-devices` command line switch.
>
> > [](../_images/ib-network-info.png)

The above image displays a congestion event. InfiniBand network information is used for displaying node and switch names instead of LIDs.

### Network Interface Controller (NIC) Profiling

#### NVIDIA NIC Metric Sampling

**Overview**

NVIDIA ConnectX smart network interface cards (smart NICs) offer advanced hardware offloads and accelerations for network operations. Viewing smart NICs metrics, on Nsight Systems timeline, enables developers to better understand their application’s network usage. Developers can use this information to optimize the application’s performance.

**Limitations/Requirements**

- NIC metric sampling supports NVIDIA ConnectX boards starting with ConnectX 5

- NIC metric sampling is supported on Linux x86_64 and Arm Server (SBSA) machines only, having minimum Linux kernel 4.12 and minimum MLNX_OFED 4.1. You can download the latest OFED driver through the DOCA-Host package as doca-ofed at [NVIDIA DOCA Downloads](https://developer.nvidia.com/doca-downloads). For archived versions of the OFED driver you can visit [MLNX_OFED Download Center](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/). If collecting NIC metrics within a container, make sure that the container has access to the driver on the host machine. To check manually if OFED is installed and get its version you can run:

  - `/usr/bin/ofed_info`

  - `cat`` ``/sys/module/"$(cat`` ``/proc/modules`` ``|`` ``grep`` ``-o`` ``-E`` ``"^mlx._core")"/version`

For the high frequency metrics, the following requirements must be met:

- The NICs must be ConnectX-7, BlueField 3 or newer.

- The NICs must have firmware XY.43.1000 or newer.

- The mlx5_fwctl module must be loaded.

- The DOCA telemetry (libdoca_telemetry.so.2) and common (libdoca_common.so.2) libraries must be installed.

- The user must have elevated privileges.

- All NICs on the target machine must have the same type of clock, Real Time Clock (RTC) or Free Running Clock (FRC). The clock can be set through the NIC’s firmware.

To check if the target system meets the requirements for NIC metrics collection you can run `nsys`` ``status`` ``--network`.

**Collecting NIC Metrics Using the Command Line**

To collect NIC performance metrics, using Nsight Systems CLI, add the `--nic-metrics` command line switch:

``` cpp
nsys profile --nic-metrics=lf my_app
```



Note

The high frequency option, `hf`, collects samples at a higher frequency compared to the `lf` option. `--nic-metrics=hf` will not collect counters for RoCE, IPoIB traffic and the Send waits metric.

**Available Metrics**

- **Bytes sent** - Number of bytes sent through the NIC port.

- **Bytes received** - Number of bytes received by the NIC port.

- **Average sent packet size** - Average byte size of packets sent through the NIC port.

- **Average received packet size** - Average byte size of packets received by the NIC port.

- **CNPs sent** - Number of congestion notification packets sent by the NIC.

- **CNPs received** - Number of congestion notification packets received and handled by the NIC.

- **Send waits** - The number of ticks during which the port had data to transmit but no data was sent during the entire tick (either because of insufficient credits or because of lack of arbitration)

Note

The counters for RoCE traffic reflect the sum of unicast and multicast traffic.

**Usage Examples**

- The `Bytes`` ``sent/sec` and the `Bytes`` ``received/sec` metrics enables identifying idle and busy NIC times.

  - Developers may shift network operations from busy to idle times to reduce network congestion and latency.

  - Developers can use idle NIC times to send additional data without reducing application performance.

- CNPs (congestion notification packets) received/sent and Send waits metrics may explain network latencies. A developer seeing the time periods when the network was congested may rewrite his algorithm to avoid the observed congestions.

#### Amazon AWS EFA NIC Metrics

Nsight Systems can now periodically sample performance counters for AWS Elastic Fabric Adapters (EFAs) and plot it on the timeline in the GUI. This enables developers to analyze how network communications may be involved with the critical path of their multi-node application. Created in collaboration with AWS, this plugin will work on [AWS EC2 NVIDIA GPU accelerated compute instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types/) .

To enable the AWS EFA metrics add the following option to the nsys `profile` or `start` commands:

``` cpp
--enable efa_metrics[,arg1[=value1],arg2[=value2], ...]
```

There are no spaces following `efa_metrics` plugin name. It is followed by a comma separated list of arguments or argument=value pairs. Arguments with spaces should be enclosed in double quotes.

Supported arguments are:

| Name | Possible Parameters | Default | Switch Description |
|----|----|----|----|
| `-efa-non-rdma` | true, false | false | Sample Infiniband non-RDMA counters |
| `-efa-sysfs` | \<path\> | /sys/class/infiniband | Root directory for EFA counters sysfs |
| `-efa-work-requests` | true, false | false | Sample Infiniband WorkRequest counters |
| `-errors` | true, false | false | Sample error counters |
| `-freq` | integer, a negative value means 1/F frequency | 10 | Target sample frequency in hertz |
| `-mode` | throughput, delta, total | throughput | Report sampled counters as a value per second, delta since previous sample, or an accumulated sum. |
| `-packets` | true, false | false | Sample packet counters |

**Usage Examples**

- `nsys`` ``profile`` ``--enable`` ``efa_metrics`` ``...`  
  Sample all EFA adapters, display as bytes per second.

- `nsys`` ``profile`` ``--enable`` ``efa_metrics,-packets,-errors,-efa-non-rdma`` ``...`  
  Sample all available EFA adapter counters.

- `nsys`` ``profile`` ``--enable`` ``efa_metrics,-mode=total`` ``...`  
  Sample all EFA adapters, display as total value sum since profiling start.

- `nsys`` ``profile`` ``--enable`` ``efa_metrics,-efa-counters-sysfs="/mnt/nv/sys",`` ``...`  
  Look for EFA counters in a different sysfs directory. Useful in some k8s environments.

This collector is the first use case for the [Nsight Systems Plugins](#nsight-systems-plugins) system.

### Network Switch Profiling

#### InfiniBand Switch Metric Sampling

NVIDIA Quantum InfiniBand switches offer high-bandwidth, low-latency communication. Viewing switch metrics, on Nsight Systems timeline, enables developers to better understand their application’s network usage. Developers can use this information to optimize the application’s performance.

**Limitations/Requirements**

IB switch metric sampling supports all NVIDIA Quantum switches. The user needs to have permission to query the InfiniBand switch metrics.

To check if the current user has permissions to query the InfiniBand switch metrics, check that the user have permission to access `/dev/infiniband/umad*`

To give user permissions to query InfiniBand switch metrics on RedHat systems, follow the directions at [RedHat Solutions](https://access.redhat.com/solutions/5929621).

To collect InfiniBand switch performance metric, using Nsight Systems CLI, add the `--ib-switch-metrics-devices` command line switch, followed by a comma separated list of InfiniBand switch GUIDs. For example:

``` cpp
nsys profile --ib-switch-metrics-devices=<IB switch GUID> my_app
```

To get a list of InfiniBand switches, reachable by a given NIC, use:

``` cpp
sudo ibswitches -C <nic name>
```



**Available Metrics**

- **Bytes sent** - Number of bytes sent through all switch ports

- **Bytes received** - Number of bytes received by all switch ports

- **Send waits** - The number of ticks during which switch ports, selected by  
  PortSelect, had data to transmit but no data was sent during the entire tick (either because of insufficient credits or of lack of arbitration)

- **Average sent packet size** - Average sent InfiniBand packet size

- **Average received packet size** - Average received InfiniBand packet size

#### InfiniBand Switch Congestion Events

##### Overview

NVIDIA Quantum InfiniBand switches offer high-bandwidth, low-latency communication.

When a switch egress port is congested, packets wait in the egress port queue before being sent out of the switch. This increases the latency of these packets.

Nsight Systems Workstation Edition gives you the ability to view when switch egress ports are congested on the Nsight Systems timeline. This enables developers to better understand latencies that are caused by the application’s network usage. Developers can use this information to optimize the application’s performance.

##### Limitations/Requirements

IB switch congestion events support requires:

- Quantum 2 switch or newer

- Having firmware version 31.2012.1068 or higher

- User need to have permission to send management datagrams

To get a list of InfiniBand switches, reachable by a given NIC, use: `sudo`` ``ibswitches`` ``-C`` ``<nic`` ``name>`

To check if the current user has permissions to send management datagrams, check that the user have permission to access `/dev/infiniband/umad*`

To give user permissions to query InfiniBand switch congestion events on RedHat systems, follow the directions at [RedHat Solutions](https://access.redhat.com/solutions/5929621).

##### Using the Command Line

To collect InfiniBand switch congestion events, using Nsight Systems CLI, add the following command line switches:

- `ib-switch-congestion-devices` This should be followed by a comma separated list of InfiniBand switch GUIDs, from which congestion events will be collected.

- `ib-switch-congestion-nic-device` This should be followed by the name of the NIC (HCA) through which InfiniBand switches will be accessed. The profiled InfiniBand switches should be reachable by this NIC.

- `ib-switch-congestion-percent` This defines the percent of InfiniBand switch congestion events to be collected. This option enables reducing the network bandwidth consumed by reporting congestion events. Values are in the \[1,100\] range.

- `ib-switch-congestion-threshold-high` This defines the high threshold for InfiniBand switch egress port queue size. When a packet enters an InfiniBand switch, its data is stored at an ingress port buffer. A pointer to the packet’s data is inserted into the egress port’s queue, from which the packet will be exiting the switch. At that point, the threshold given by this command switch is compared to the egress queue data size. If the queue data size exceeds the threshold, a congestion event is reported. The threshold is given in percent of the ingress port size. An egress port queue can point to data coming from multiple ingress port buffers, therefore the threshold can be bigger than 100%. Values are in the (1,1023\] range



### Network Interface Device Profiling

Network Interface Devices (NIDs) are devices (usually ISP-owned) separating public and private networks. Nsight Systems can now periodically sample performance counters for network interface devices and plot them on the timeline in the GUI.

To enable the network devices metrics add the following option to the nsys `profile` or `start` commands:

``` cpp
--enable network_interface[,arg1[=value1],arg2[=value2], ...]
```

There are no spaces following `network_interface` plugin name. It is followed by a comma separated list of arguments or argument=value pairs. Arguments with spaces should be enclosed in double quotes.

Supported arguments are:

| Short name | Long name | Possible Parameters | Default | Switch Description |
|----|----|----|----|----|
| `-i` | `--interval` | integer | 100000 | Sampling interval in microseconds |
| `-d` | `--devices` | regular expression | “.+” (and filtering for physical devices) | Device(s) to sample |
| `-m` | `--metrics` | regular expression | “.\*\_bytes” | Metric(s) to sample |
| `-h` | `--help` |  |  | Print help message |

**Usage Examples**

- `nsys`` ``profile`` ``--enable`` ``network_interface`` ``...`  
  Sample bytes metrics for all physical network devices every 100ms.

- `nsys`` ``profile`` ``--enable`` ``network_interface,-dall`` ``...`  
  Sample bytes metrics for all network devices every 100ms.

- `nsys`` ``profile`` ``--enable`` ``network_interface,-i10000,-dall,-m".+"`  
  Sample all metrics, for all network devices, every 10ms.

For general information on Nsight Systems plugins please refer to [Nsight Systems Plugins](#nsight-systems-plugins) system.

## Network Storage Profiling

Nsight Systems can profile several major storage / remote storage protocols. It also ships with the `storage_util_map` and `file_access_sum` recipes for post-collection analysis. See [Post-Collection Analysis Guide](../AnalysisGuide/index.html#id1)

Note

Storage metrics profiling requires additional kernel support and is tailored for HPC systems. This feature is not relevant to Nsight Systems Embedded Platforms Edition.

To activate this feature, use the Nsight Systems CLI `--storage-metrics` option, followed by a comma-separated list of the desired arguments.

**Available arguments:**

- `--nfs-volumes={all`` ``|`` ``volume1[,volume2][,volume3..]}`: enable NFS storage profiling for the specified volume(s) (specify `all` to profile all volumes).

- `--lustre-volumes={all`` ``|`` ``volume1[,volume2][,volume3..]}`: enable Lustre storage profiling for the specified volume(s) (specify `all` to profile all volumes).

- `--lustre-llite-dir=<path>`: specifies the path of the llite directory mount. This is the `/sys/kernel/debug/lustre/llite` directory mount point (mandatory if Lustre profiling is enabled).

- `--storage-devices={all`` ``|`` ``device1[,device2][,device3..]}`: enable storage profiling of the specified local storage or NVMe-oF device(s) (specify `all` to profile all devices).

- `--interval=<value>`: sampling interval in milliseconds. Valid range is 1-60000 (default: 1).

- `--cache-samples=<value>`: number of samples to cache before submitting the events. Valid range is 1-1000 (default: 100).

**Usage Examples**



In the report file, under ‘Timeline view’, the storage metrics can be viewed in the **Mounts** section. Each row contains metrics for one volume or device, with the storage type next to the volume / device name. Expanding each row will show the collected metrics for that volume / device.



The `stdout` and `stderr` log files for the storage metrics collection process can be viewed under the ‘Files’ section, which may assist in debugging.

It is also possible to use combinations of these arguments to profile multiple storage protocols at once. For example:

``` bash
./nsys profile --storage-metrics --nfs-volumes=all,--lustre-volumes=all,--storage-devices=<device_name1>,<device_name2>,--lustre-llite-dir=<path_to_llite_directory> <target-application>
```

Note

There are two types of Read/Write metrics:

Application-level Read/Write - Displays **quantities** of data read/written to the storage device **by applications** (in Bytes).

Driver-level Read/Write - Displays **throughput** of data read/written to the storage device **by the driver** (in bytes/s).

For example, when an application uses the “write” POSIX function to write 10 MB of data into a file, the entire 10 MB will appear, in a single sampling point, at the Application-level Write counter. The same 10 MB of data may be spread across multiple Driver-level Write counter sampling points, since it may take a bit of time for the NFS driver to write 10 MB of data into the NFS storage server.

### NFS volumes counters

Example Nsight Systems command line for NFS storage profiling:

``` bash
./nsys profile --storage-metrics --nfs-volumes=all <target-application>
```



### Lustre volumes counters

Example Nsight Systems command line for Lustre storage profiling:

``` bash
./nsys profile --storage-metrics --lustre-volumes=dtdata_test,--lustre-llite-dir=/mnt/lustre-stats/llite <target-application>
```



**Exposing Lustre driver counters to non-privileged users**

The Lustre driver exposes performance counters via virtual files residing under `/sys/kernel/debug/lustre`. However, this path is not accessible to non-privileged users.

To expose the Lustre counters to non-privileged users, a superuser should create a mount point to `/sys/kernel/debug/lustre`. For example:

``` bash
su - root
mkdir /mnt/lustre-stats
mount --bind /sys/kernel/debug/lustre /mnt/lustre-stats
```

The `--lustre-llite-dir=` command line argument should point to the `llite` directory under this mount point; this will enable Nsight Systems to read the Lustre counters. For example: `--lustre-llite-dir=/mnt/lustre-stats/llite`

### Local and NVMe-oF volumes counters

Example Nsight Systems command line for local storage and NVMe-oF device profiling:

``` bash
./nsys profile --storage-metrics --storage-devices=all <target-application>
```



### S3 Trace

Nsight Systems can capture information about Amazon S3 storage operations performed by the profiled process. When S3 tracing is enabled, upload and download activity is recorded on the timeline, along with metadata such as bucket name, object key, bytes transferred, and operation result.

In addition, Nsight Systems aggregates S3 trace events to produce per-process statistics including upload and download throughput and average transfer sizes.

S3 trace is available on Linux targets only.

The following S3 client libraries are supported:

- **AWS CRT** — Native C/C++ applications (or higher-level SDKs) that use the AWS Common Runtime S3 client (`aws-c-s3`). Nsight Systems traces S3 upload and download request operations, and the individual HTTP transactions within each request.

  Note

  Requires aws-c-s3 version 0.9.3 or newer for tracing S3 operations. HTTP transaction tracing requires aws-c-s3 version 0.10.0 or newer. The profiled process must also dynamically link with the CRT library.

- **AWS CPP SDK** — C++ applications that use the AWS C++ SDK (`aws-cpp-sdk`) for S3 operations can be traced. Nsight Systems traces both the S3 Client and the S3Crt Client.

  Note

  Tracing S3Client requires aws-cpp-sdk version 1.11.0 or newer. Tracing S3CrtClient also requires aws-c-s3 version 0.9.3 or newer (bundled in aws-cpp-sdk version 1.11.715 or newer).

  The profiled process must dynamically link with the CPP SDK library.

- **Boto3** — Python applications that use the `boto3` library for S3 operations can be traced. The following operations are traced across Client, Bucket, and Object S3 resource types:

  ``` cpp
  upload_file              download_file
  upload_fileobj           download_fileobj
  put_object               get_object
  head_object              upload_part
  create_multipart_upload  abort_multipart_upload
  generate_presigned_url   generate_presigned_post
  ```

- **S3TorchConnector** — Python applications that use the `s3torchconnector` library for PyTorch dataset access over S3 (e.g., `S3IterableDataset`, `S3MapDataset`, Checkpoint and direct reader/writer operations).

- **Tensorflow-io** — Python applications that use the `tensorflow-io` library to access S3 objects via `tf.io.gfile.GFile` with `s3://` paths.

#### Usage Example

Example trace with Boto3:



To enable S3 tracing from Nsight Systems:

**CLI** — Use the `-t`, `--trace` option with the `s3` or `s3-verbose` parameter. See [Command Line Options](index.html#command-line-options) for more information.

``` bash
nsys profile --trace=s3 <application> [application-arguments]
```

Two levels of detail are available:

- `--trace=s3` — Collects S3 operation ranges, with core attributes, including bytes transferred, bucket name, key name, file path, and result status.

- `--trace=s3-verbose` — In addition to everything collected by `s3`, Nsight Systems will collect additional per-request metadata and breakdown of individual HTTP transactions. This additional data is intended for in-depth low-level analysis of transfer behavior, but may increase trace volume and processing overhead. This mode only affects tracing of AWS CRT, AWS CPP SDK, and Boto3 applications.



#### Process Statistics

When S3 trace data is collected, Nsight Systems aggregates the events from the profiled process to display the following statistics in the timeline:

- **Download Throughput** — Aggregate download throughput over time (bytes/s).

- **Upload Throughput** — Aggregate upload throughput over time (bytes/s).

- **Avg Download Size** — Average file/object size of currently active download operations (bytes).

- **Avg Upload Size** — Average file/object size of currently active upload operations (bytes).

These statistics can help identify periods of high or low S3 activity and reveal bottlenecks in data transfer patterns.



### GDS (GPUDirect Storage) Trace

NVIDIA GPUDirect Storage (GDS) enables direct memory access (DMA) between storage and GPU memory. This avoids a bounce buffer through the CPU, increasing storage access bandwidth and decreasing latency and utilization load on the CPU. Information about GDS can be found at [NVIDIA Magnum IO GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/).

Nsight Systems can capture information about GDS, specifically the various cuFile API calls made by the profiled process. GDS profiling is supported on Linux x64 and SBSA operating systems.



Note

Before collecting GDS metrics, ensure that **NVIDIA GPUDirect Storage** is installed correctly on your system. For installation instructions, refer to the [NVIDIA GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#nvidia-gpudirect-storage-installation-and-troubleshooting-guide/).

You can validate that GDS is installed correctly by running the gdscheck.py tool:

``` bash
/usr/local/cuda/gds/tools/gdscheck.py -p
```

The tool should confirm that the intended filesystem type is supported, and that platform verification has passed successfully.

### GDS (GPUDirect Storage) Counters

Nsight Systems can collect GDS user-space metrics from profiled processes. GDS metrics collection is supported on Linux x64 and SBSA operating systems.

Note

This is only supported with GPUDirect Storage v1.16.0 or newer, which is available from CUDA Toolkit v13.1.

**Available arguments:**

- `--gds-metrics`: Enable GDS (GPUDirect Storage) user-space performance metrics collection.

- `--gds-libs-path=<path>`: Specify a directory containing GPUDirect Storage libraries (must contain libcufile.so). Use this argument if the GDS libraries are located in a different path than the default. Default is `/usr/local/cuda/lib64`. This argument is used together with `--gds-metrics`.

**Usage Example**

To profile a process with GDS metrics:

`./nsys`` ``profile`` ``--gds-metrics`` ``<target-application>`

If your GDS libraries are installed in a custom location:

`./nsys`` ``profile`` ``--gds-metrics`` ``--gds-libs-path=/custom/path/to/gds/libs`` ``<target-application>`



### SCADA Metrics Profiling

NVIDIA SCADA (SCaled Accelerated Data Access) is a storage I/O architecture where GPUs directly initiate and control storage operations, removing CPU involvement from both the control path and the data path. It is optimized for AI workloads that require high-throughput, fine-grained accesses (\< 4 KB) from thousands of parallel GPU threads to NVMe storage. A SCADA deployment consists of a client library on the GPU side and a server daemon that manages the NVMe drives.

Nsight Systems can periodically sample performance counters and histograms from a running SCADA server and display them on the timeline in the GUI. This enables developers to monitor SCADA server activity and correlate it with GPU and CPU events.

SCADA metrics profiling is supported on Linux x64 and SBSA operating systems.

#### Enabling SCADA Metrics Collection

To enable SCADA metrics collection, add the following option to the nsys `profile` or `start` commands:

``` cpp
--enable scada_metrics[,--sampling-frequency=<value>]
```

There are no spaces following the `scada_metrics` plugin name. It is followed by a comma-separated list of arguments or argument=value pairs.

Note

A SCADA server must be running on the target system before starting the profiling session. The plugin communicates with the server through a Unix Domain Socket at `/tmp/scada_profiler_socket`. If the server is not running, the plugin will report an error and exit.

Supported arguments are:

| Name | Possible Values | Default | Description |
|----|----|----|----|
| `-s` / `--sampling-frequency` | 1 – 1000 | 1000 | Sampling frequency in Hz. Controls how often the plugin requests a new metrics sample from the SCADA server. |

**Usage Examples**

- `nsys`` ``profile`` ``--enable`` ``scada_metrics`` ``<target-application>`  
  Collect SCADA metrics at the default sampling frequency of 1000 Hz.

- `nsys`` ``profile`` ``--enable`` ``scada_metrics,-s,100`` ``<target-application>`  
  Collect SCADA metrics at 100 Hz.

- `nsys`` ``profile`` ``--enable`` ``scada_metrics,--sampling-frequency=10`` ``<target-application>`  
  Collect SCADA metrics at 10 Hz (one sample every 100 ms).

For general information on Nsight Systems plugins please refer to [Nsight Systems Plugins](#nsight-systems-plugins) system.

#### Viewing SCADA Metrics in the Report

In the report file, under **Timeline view**, SCADA metrics appear in the **SCADA Metrics** section. Each counter is plotted as a time-series graph and each histogram is displayed with its bucketed distribution over time.



The `stdout` and `stderr` log files for the SCADA metrics collection process can be viewed under the **Files** section, which may assist in debugging connectivity or sampling issues.



#### Available Metrics

The specific metrics collected by the plugin are determined entirely by the SCADA server. At the start of each profiling session, the plugin fetches a **metrics schema** from the server that describes the available metrics — their names, units, and histogram bin definitions. Different SCADA server configurations may expose different sets of metrics.

There are two types of metrics:

- **Counters** — Scalar values sampled at each collection interval (e.g. total received buffers, average latency). Each counter is plotted as a time-series graph on the timeline.

- **Histograms** — Distribution metrics with predefined bins (e.g. latency distribution, commands-per-buffer distribution). Each histogram is displayed with its bucketed distribution over time.

The following screenshots show example metrics from a particular server configuration:

**Counters**



**Histograms**



## Python Profiling

Nsight Systems has several features that have been added in the last few years to enhance users optimizing their python code.

Note

You may find that all of your python application output comes at the end of the run instead of as events happen.

Python will change the buffering of stdout depending on whether it points to a tty or something else. Nsight Systems redirects the application stdout to a pipe to demultiplex stdout to both a file and the terminal. As a side effect, it makes Python change stdout buffering from line-buffered to page-buffered. You can use `python`` ``-u` option or the `PYTHONUNBUFFERED`` ``environment` variable to override this behavior.

### Python Backtrace Sampling

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of periodically capturing Python backtrace information. This functionality is available when tracing Python interpreters of version 3.9 or later. Capturing Python backtrace is done in periodic samples, in a selected frequency ranging from 1Hz - 2KHz with a default value of 1KHz. Note that this feature provides meaningful backtraces for Python processes, when profiling Python-only workflows, consider disabling the CPU sampling option to reduce overhead.

In Nsight Systems GUI, Python backtrace sampling is visualized similar to CPU backtrace sampling. See also [Visualizing CPU Profiling Results](index.html#visualizing-cpu-profiling-results).

To enable Python backtrace sampling from Nsight Systems:

**CLI** — Set `--python-sampling=true` and use the `--python-sampling-frequency` option to set the sampling rate.

**GUI** — Select the **Collect Python backtrace samples** checkbox.

> 

Example screenshot:



### Python Functions Trace

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of using NVTX to annotate Python functions.

The Python source code does not require any changes. This feature requires CPython interpreter, release 3.8 or later.

The annotations are configured in a JSON file. An example file is located in Nsight Systems installation folder in `<target-platform-folder>/PythonFunctionsTrace/annotations.json`.

For PyTorch applications, Nsight Systems provides a predefined annotations file located in `<target-platform-folder>/PythonFunctionsTrace/pytorch.json`.

For Dask applications, Nsight Systems provides a predefined annotations file located in `<target-platform-folder>/PythonFunctionsTrace/dask.json`.

Note

Annotating a function from the module `__main__` is not supported.

To enable Python functions trace from Nsight Systems:

**CLI** — Set `--python-functions-trace=<json_file>`.

**GUI** — Select the **Python Functions trace** checkbox and specify the JSON file.

> 

Example screenshot:



### Python GIL Tracing

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of tracing when Python threads are waiting to hold and holding the GIL (Global Interpreter Lock).

The Python source code does not require any changes. This feature requires CPython interpreter, release 3.9 or later.

This feature is not supported on Python that was compiled with `Py_GIL_DISABLED=1` (See [Python documentation](https://docs.python.org/3/howto/free-threading-extensions.html#identifying-the-free-threaded-build-in-c) for details).

**CLI** — Set `--trace=python-gil`.

**GUI** — Select the **Trace GIL** checkbox under **Python profiling options**.

> 

Example screenshot:



### PyTorch Profiling

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of automatically annotating common PyTorch operations with execution time ranges.

The Python source code does not require any changes. This feature requires CPython interpreter, release 3.8 or later.

To enable PyTorch [autograd nvtx](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx), run Nsight Systems from the CLI using the `--pytorch` option:

Set `--pytorch=autograd-nvtx` for enabling `torch.autograd.profiler.emit_nvtx(record_shapes=False)` or `--pytorch=autograd-shapes-nvtx` for enabling `torch.autograd.profiler.emit_nvtx(record_shapes=True)` (implies `--trace=nvtx`).

Set `--pytorch=functions-trace` for automatically annotating PyTorch operations like [forward operations](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward), [backward operations](https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward), [step operations](https://docs.pytorch.org/docs/stable/optim.html#taking-an-optimization-step), etc. with execution time ranges.

Set `--pytorch=functions-trace-shapes` to attach additional data to each annotated range, such as tensor shapes, training parameters, etc. `functions-trace` and `functions-trace-shapes` are mutually exclusive.

Both `--pytorch=functions-trace` and `--pytorch=functions-trace-shapes` imply `--python-functions-trace=<nsys_install_dir>/<target-arch>/PythonFunctionsTrace/pytorch.json`.

`autograd-nvtx` and `autograd-shapes-nvtx` options can be combined with `functions-trace` or `functions-trace-shapes` by adding them separated by a comma.

When profiling a [vLLM](https://docs.vllm.ai/en/stable/) application, the `functions-trace` option also automatically annotates vLLM-specific methods such as `AsyncLLM.generate()`, `Worker.execute_model()`, and `Worker.sample_tokens()` under a dedicated *vLLM* NVTX domain. These annotations include correlated request IDs for tracking individual requests across the inference pipeline.

Example screenshots:

 

### Dask Profiling

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of automatically annotating common Dask functions with execution time ranges.

The Python source code does not require any changes. This feature requires CPython interpreter, release 3.8 or later.

Set `--dask=functions-trace` for enabling Dask functions trace. This option sets `--python-functions-trace=<nsys_install_dir>/<target-arch>/PythonFunctionsTrace/dask.json` and will rename relevant threads to ‘Dask Worker’ and ‘Dask Scheduler’.

`dask.json` can be modified to include additional functions to be traced from any Python module.

Example screenshot:



## Profiling with DRIVE Hypervisor

Nsight Systems and DRIVE Hypervisor support periodic CPU sampling with call stacks. It works both on DRIVE Linux and QNX.

The call stacks are collected using frame pointers. The Linux kernel, QNX kernel, and user space libraries provided by NVIDIA are compiled with frame pointers. To ensure correct call stacks, we recommend compiling all application code with frame pointer support, using `-fno-omit-frame-pointer` with GCC, Clang, and QCC.

This is an experimental feature and is expected to change in the future.

**The symbols can be resolved both for user space code, and for kernel space code:**

- In the user space, the Cross-Hypervisor (XHV) sampling events are matched with the CPU thread state trace coming from Linux Perf and QNX Tracelogger. After that, Nsight Systems can know the module filename, and can resolve symbols directly from these files if they are unstripped, or by looking up additional files with symbols. See more details below.

- In the kernel space (Linux kernel, QNX kernel, and additional service VMs), the symbols are resolved using the ELF file with symbols specified. `kernel_symbols.json` input file specifies the location of this ELF file.

**Please follow the steps below to learn how to:**

- Flash the devkit (these steps are given just as an example, the exact steps might differ in your case).

- Copy the necessary files: pct.json, eventlib schema files, and kernel_symbols.json.

- Compose kernel_symbols.json to allow resolving symbols in the Linux kernel, QNX kernel, and additional service VMs.

- See example CLI commands to collect data.

**Known issues:**

- At the moment, this feature is not compatible with standard CPU sampling on Linux and QNX.

- When enabled together, hypervisor trace plus XHV sampling can write too much data into the same eventlib buffers, and the Nsight Systems agent might not be able to keep up with the rate, losing events. If that happens, please disable hypervisor trace events with `--xhv-trace-events=none`.

**Flashing DRIVE OS QNX/Linux**

Log into the NVIDIA GPU Cloud (NGC):

``` text
sudo docker login nvcr.io

Username: ``$oauthtoken``
Password: <NGC API key>
```

Docker command:

``` cpp
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it <docker image>
```

`<docker`` ``image>` - docker image link.

Examples:

6.0.8.0 QNX:

``` cpp
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it nvcr.io/{MY_NGC_ORG}/driveos-pdk/drive-agx-orin-qnx-aarch64-pdk-build-x86:6.0.8.0-0003
```

6.0.9.1 QNX:

``` cpp
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it nvcr.io/{MY_NGC_ORG}/driveos-pdk/drive-agx-orin-qnx-aarch64-pdk-build-x86:6.0.9.1-latest
```

6.0.8.0 Linux:

``` cpp
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it nvcr.io/{MY_NGC_ORG}/driveos-pdk/drive-agx-orin-linux-aarch64-pdk-build-x86:6.0.8.0-0003
```

Inside of container, flash with flash.py:

``` cpp
cd /drive
./flash.py <aurix> <board>
```

- `<board>` - target board base name: ‘p3710’ or ‘p3663’.

- `<aurix>` - Aurix serial port, for example: /dev/ttyACM1, /dev/ttyUSB1.

Examples:

Firespray p3710:

``` cpp
./flash.py /dev/ttyACM1 p3710
```

Drive Orin p3663:

``` cpp
./flash.py /dev/ttyUSB1 p3663
```

List the available EMMC and UFS partitions:

``` cpp
df -h
```

Format a power-safe file system partition, and mount it, example for `vblk_ufs40`:

``` cpp
mkqnx6fs /dev/vblk_ufs40 -q
mount -o rw /dev/vblk_ufs40 /
```

``` text
# df -h
/dev/vblk_ufs40             116G      7.5G      108G       7%  /
ifs                          16M       16M         0     100%  /
ifs                          52M       52M         0     100%  /
...
```

Note

For more information about DRIVE OS installation, see the following link: [NVIDIA DRIVE OS Documentation](https://docs.nvidia.com/drive) (useful pages: **DRIVE OS Linux Installation Guide**, **DRIVE OS QNX Installation Guide**).

**Create XHV Directory**

Inside of container, examples for p3710, QNX/Linux:

QNX:

``` cpp
cd /drive_flashing
mkdir -p xhv/hypervisor/configs/t234ref-release/pct/qnx xhv/schemas
cp -rv /drive/drive-foundation/virtualization/hypervisor/t23x/configs/t234ref-release/pct/p3710-10-a03/qnx/pct.json ./xhv/hypervisor/configs/t234ref-release/pct/qnx/
cp -rv /drive/drive-foundation/schemas/event ./xhv/schemas/
```

Linux:

``` cpp
cd /drive_flashing
mkdir -p xhv/hypervisor/configs/t234ref-release/pct/linux xhv/schemas
cp -rv /drive/drive-foundation/virtualization/hypervisor/t23x/configs/t234ref-release/pct/p3710-10-a03/linux/pct.json ./xhv/hypervisor/configs/t234ref-release/pct/linux/
cp -rv /drive/drive-foundation/schemas/event ./xhv/schemas/
```

Example of XHV directory (Linux):

``` text
xhv/
├── hypervisor
│             └── configs
│                 └── t234ref-release
│                     └── pct
│                         └── linux
│                             └── pct.json
└── schemas
    └── event
        ├── audioserver_events.json
        ├── bpmp_events.json
        ├── cem_events.json
        ├── hv_events.json
        ├── i2c_events.json
        ├── Makefile.gen-event-headers.tmk
        ├── monitor_events.json
        ├── se_events.json
        ├── sysmgr_events.json
        └── vsc_events.json
```

Copy XHV directory to target:

``` text
scp -r xhv <user>@<target-IP>
```

eventlib_dump tool (QNX/Linux):

``` cpp
cp -rv /drive/drive-qnx/nvidia-bsp/aarch64le/sbin/eventlib_dump /drive_flashing/
cp -rv /drive/drive-linux/filesystem/contents/bin/eventlib_dump /drive_flashing/
```

**Specific Command Line Options**

| Option | Possible Parameters | Default | Switch Description |
|----|----|----|----|
|  |  |  |  |
| `--sample` | process-tree, system-wide, xhv, xhv-system-wide, none | process-tree | Select ‘xhv’ or ‘xhv-system-wide’ to enable Cross-Hypervisor (XHV) sampling, requires root privileges. |
| `--xhv-vm-symbols` | \< filepath kernel_symbols.json \> | none | XHV sampling config (optional, for kernel symbols). |
| `--xhv-trace` | \< filepath pct.json \> | none | Collect hypervisor trace. |
| `--xhv-trace-events` | all, none, core, sched, irq, trap | all | HV trace events. |

Examples:

``` cpp
nsys profile --sample=xhv --trace=nvtx,osrt,cuda --xhv-vm-symbols=/root/kernel_symbols.json --xhv-trace=/root/xhv/hypervisor/configs/p3710-10-a01/pct/qnx/pct.json --xhv-trace-events=none sleep 5
nsys profile --sample=xhv-system-wide --xhv-vm-symbols=/root/kernel_symbols.json --xhv-trace=/root/xhv/hypervisor/configs/p3710-10-a01/pct/qnx/pct.json --xhv-trace-events=none sleep 5
```

Example screenshot:



**Config File (for kernel symbols)**

Examples:

QNX, `kernel_symbols.json` file:

``` json
{
    "guest_cfg": [
        {
            "guest_id": 0,
            "guest_name": "Guest VM 0",
            "symbols": "/root/symbols/procnto-smp-instr-safety.guest_vm.bin.sym"
        },
        {
            "guest_id": 1,
            "guest_name": "Update service",
            "symbols": "/root/symbols/procnto-smp-instr-safety.update_vm.bin.sym"
        },
        {
            "guest_id": 2,
            "guest_name": "Resource Manager Server"
        },
        {
            "guest_id": 3,
            "guest_name": "Storage Server"
        },
        {
            "guest_id": 4,
            "guest_name": "Ethernet Server"
        },
        {
            "guest_id": 5,
            "guest_name": "Debug Server"
        }
    ],
    "symbol_files": {
        "Sidekick": "/root/symbols/sidekick.unstripped"
    }
}
```

Linux, `kernel_symbols.json` file:

``` json
{
    "guest_cfg": [
        {
            "guest_id": 0,
            "guest_name": "Guest VM 0",
            "symbols": "/home/nvidia/vmlinux"
        },
        {
            "guest_id": 1,
            "guest_name": "Update service"
        }
    ],
    "symbol_files": {
    }
}
```

**Symbol Files**

The list of directories with symbol files:

- CLI: `DbgFileSearchPath` config option, for example: `DbgFileSearchPath="/lib:/root/symbols"` - list of directories with symbol/debug files. On Linux, the default path is `/usr/lib/debug`. On QNX, there is no default path.

  Example:

  ``` text
  NSYS_CONFIG_DIRECTIVES='DbgFileSearchPath="/lib:/root/symbols"' nsys profile --sample=xhv  --xhv-vm-symbols=/root/kernel_symbols.json --xhv-trace=/root/xhv/hypervisor/configs/p3710-10-a01/pct/qnx/pct.json --xhv-trace-events=none sleep 5
  ```

- GUI: `Symbol`` ``location` button.

The search is non-recursive.

There are several ways of searching for symbol files - Nsight Systems tries them sequentially for each target file:

- Build-id debug files (CLI only)

  \<symbol directory\>/.build-id/… - directories with debug files (or links to debug files).

  Example:

  ``` text
  .build-id/
  ├── 00
       └── 6627b119cc2aee77e10e0535fc243fce8fe66e.debug
  ├── 01
       ├── 3e4007e3cb24359203fc02b63bb90f16db5b23.debug
       └── fb938bc0f029c41a8e1e88f01f88f75cf3a0d3.debug
  ...
  ```

- Debuglink files (CLI only)

  \<symbol directory\>/\<symbol file\> - both filename and CRC from debuglink section must be matched for the symbol file.

- File name and build-id (CLI/GUI)

  \<symbol directory\>/\<symbol file\> - by filename and build-id.

**XHV profiling from the GUI**

XHV options:



Use this dialog to specify XHV parameters:

- `Collect`` ``HV`` ``Trace` - Enable XHV tracing.

- The location of `pct.json` file on the host. There is predefined hierarchy of XHV JSON files, for example:

``` text
xhv/
├── hypervisor
│             └── configs
│                 └── t234ref-release
│                     └── pct
│                         └── linux
│                             └── pct.json
└── schemas
    └── event
        ...
        ├── hv_events.json
        ...
```

- `Collect`` ``VM`` ``Profile` - Enable XHV sampling, depends on `Collect`` ``HV`` ``Trace`.

- `Event`` ``mask` - Select XHV trace events, this option can be specified as `None`.

- The location of `kernel_symbols.json` file on the host. Note that this file contains target paths to the kernel symbol files (see examples above).

- `Skip`` ``idle` and `Combine`` ``EL0` checkboxes are deprecated.

## Adding Your Own Collection to a Report

Nsight Systems allows the user to add additional information to a report file for display with other Nsight Systems options.

### Nsight Systems Plugins

Nsight Systems plugins are tools that extend its data collection capabilities, available via CLI with `--enable` command option and via GUI. There are multiple locations where the Nsight Systems searches for available plugins described in the [Plugin discovery](#plugin-discovery) section.

The bundled plugins are created, documented and maintained by the Nsight Systems team.

In addition to the plugins created by the Nsight Systems team, we also have a GitHub repository of plugins created and supported by third parties. See [Third Party Plugins List](https://nvidia.github.io/NsightSystemsPlugins/) or [GitHub repository](https://github.com/NVIDIA/NsightSystemsPlugins).

Warning

Third party plugins are not tested or validated in any way by the Nsight Systems team. NVIDIA is not responsible for the content or behavior of those plugins.

#### How to launch a plugin

In CLI plugins are enabled with the `--enable` command option that also allows passing arguments to the plugin. It’s possible to launch multiple instances of the same plugin by using multiple `--enable` options.

Depending on the plugin type it may be available in `nsys`` ``profile`, `nsys`` ``start` or `nsys`` ``launch` commands. For example, a minimal plugin that only sets some environment variables will be applicable to the `nsys`` ``launch` and `nsys`` ``profile` commands if the latter launches a profiled application because otherwise there is little sense in modifying the environment. Refer to a plugin documentation to find out its supported usage pattern.

Nsight Systems plugins can be configuration-only, standalone processes, shared libraries injected into target processes, or any combination of these features. When enabled, standalone plugin processes are launched just before the data collection starts and are terminated right before the collection stops. This default behavior can be amended, see the [Developing an Nsight Systems Plugin](#developing-an-nsight-systems-plugin) section. Configuration and in-process shared libraries plugins are tied to the lifetime of a target process and are not unloaded after data collection stops.

Standalone plugin processes are launched with the same privileges as the running instance of Nsight Systems. If a plugin needs elevated privileges then Nsight Systems may need to run elevated.

#### How to pass arguments to a plugin

To pass arguments to a plugin, specify them as a part of `--enable` option after plugin name when launching the target application. The arguments should be separated by commas only (no spaces). On non-Windows platforms, commas can be escaped with a backslash `\\`, and the backslash itself can be escaped by another backslash `\\\\`. On Windows, use the caret `^` as the escape character (e.g., `^,` for a literal comma), and `^^` for a literal caret. To include spaces in an argument, enclose the argument in double quotes `"`.

See the section on the [Amazon AWS Elastic Fabric Adapter (EFA) Network Counters](#aws-efa-plugin) for an example.

#### Supported platforms

Nsight Systems plugins are supported on x86_64, arm64 Linux and x86_64 Windows.

#### Plugin discovery

Nsight Systems will search for plugins in the following locations:

> 1.  User-specified locations via environment variable.
>
> 2.  Plugins bundled with this version of the profiler.
>
> 3.  Third-party unversioned system-wide plugins.

The bundled and third-party plugins are placed in locations that should require elevated privileges for modification. This allows Nsight Systems to assume that all files in those locations are trusted and perform no additional verification.

Warning

**Security notice**: The user is responsible for ensuring security of the locations specified via environment variable. Nsight Systems cannot perform security checks and will trust and may execute any plugin that was discovered via user-provided location.

**Third-party unversioned system-wide plugins**

The paths for these plugins are platform-dependent.

> Linux: `/opt/nvidia/nsight-systems-plugins`
>
> Windows: `C:\\Program`` ``Files\\NVIDIA`` ``Corporation\\Nsight`` ``Systems`` ``Plugins`

**User-specified locations via environment variable**

User may specify multiple lookup locations via environment variable: `NSYS_PLUGIN_SEARCH_DIRS` The location must be separated by a platform-dependent separator:

> Linux: `:`
>
> Windows: `;`

**Listing available plugins**

To list all available plugins use the `nsys`` ``plugins`` ``list` command. It will collect all search locations and enumerate plugins in alphabetical order. Plugins that have failed manifest validation will have an **(error)** prefix. Note that it’s not possible to enable any plugins as long as there’s at least one that fails validation.

### Developing an Nsight Systems Plugin

#### Plugin manifest file

Plugin manifest is a file in the YAML format. It describes a plugin, its features and requirements. Nsight Systems validates all discovered manifests when at least one plugin has been enabled and will refuse to run the given command if any manifest fails the validation.

**Supported top-level manifest keys**

| Name | Required | Value type | Description |
|----|----|----|----|
| PluginName | Yes | String | A globally unique name for selecting the plugin via `--enable` command. |
| Description | Yes | String | A short description printed when listing plugins. |
| ExtendedDescription | No | String | Full description of a plugin. |
| ExtendedDescriptionForGui | No | String | Full description of a plugin shown in the GUI. |
| TargetEnvironment | No | Dictionary | Amend and possibly override the environment when launching a profiled application. Empty and duplicate keys are not allowed. |
| ExecutablePath | No | String | A path to a standalone plugin executable file. Both absolute and relative paths are supported. The manifest path is used as a base for relative executable paths. The given path must exist for a manifest to pass validation. |
| LibraryPath | No | String | A path to a shared library that will be injected into profiled processes. Both absolute and relative paths are supported. The manifest path is used a as base for relative executable paths. The given path must exist for a manifest to pass validation. |

At least one of the `ExecutablePath`, `LibraryPath`, or `TargetEnvironment` entries is required but all can be used simultaneously in any combination.

**Example manifest file**

``` yaml
PluginName: unique_name
ExecutablePath: bin/plugin
LibraryPath: libPlugin.so
TargetEnvironment: {KEY: VALUE}
Description: This is an example plugin manifest.
```

#### Plugin types

A plugin manifest may specify one or more actionable entries: `ExecutablePath`, `LibraryPath`, `TargetEnvironment`. They are named as “standalone”, “in-process”, and “configuration” plugin types. When selected with `--enable` option Nsight Systems will perform all applicable actionable entries. The “in-process” and “configuration” plugin types are only executed when starting a profiled application. The “standalone” plugin type is executed every time the report collection is started.

#### NsysDK Collector

The NsysDK Collector is an API for plugins to communicate with the profiler. It’s a header-only library that allows fetching current profiler state and extend data collection scope through the finalization stage. Documentation is embedded into source files and additionally covered in this section. The library sources are deployed with Nsight Systems installation, for example `/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsysdk`.

#### Deploying plugins

During plugin development it’s more convenient for Nsight Systems to pick it up directly from a build output location rather than copy the binary each time to the pre-defined search path. The easiest way to do this is to export the `NSYS_PLUGIN_SEARCH_DIRS` environment variable with the location of a folder that contains the plugin manifest.

#### Minimal standalone plugin

Simple standalone plugins are expected to initialize and shutdown near instantly. Nsight Systems uses SIGTERM for indicating that a plugin needs to stop producing data and exit, so it expects plugins to gracefully handle the signal and exit cleanly. The data collected from plugins is in the form of [NVTX events](index.html#nvtx-trace), stdout and stderr streams. The source code for the `network_interface` plugin is deployed as an example in `<profiler`` ``installation`` ``dir>/target-linux-x64/samples/NetworkPlugin.cpp`.

#### Deferred events standalone plugin

In some cases plugins may need to perform post-processing of the collected data or are obtaining their data from another source that only becomes available after a delay. In such cases if plugins were to immediately shutdown following a signal from the profiler some data would be lost. To avoid this, a plugin may acquire finalization tokens through the NsysDK Collector API. Once successfully obtained, these tokens will prevent profiler from stopping data collection until all tokens have been released allowing plugins to finish emitting their events. A plugin process may release finalization tokens explicitly or simply exit to have the tokens it held automatically released.

Note, that this feature is not designed for facilitating a lengthy de-initialization as the profiling data collection is still running in this state. If a plugin needs significant time to shutdown then it should release finalization tokens explicitly and handle SIGTERM signal.

The sample code below is a skeleton implementation of a plugin that utilizes finalization tokens. Compile the example with `g++`` ``-I`` ``./nvtx/include`` ``-I`` ``./nsysdk/include`` ``-ldl`` ``plugin.cpp`. On POSIX platforms the NVTX library requires adding the `-ldl` linker option. Refer to the [NVTX documentation](https://nvidia.github.io/NVTX/) to learn how to use the deferred events feature.

``` cpp
#include <nvtx3/nvToolsExt.h>
#include <nsysdk/collector.h>

#include <chrono>
#include <thread>

int main(int argc, char* argv[])
{
    // <Additional initialization>

    if (!nsysdkCollectorAcquireFinalizationToken())
        return 1; // Not launched as a plugin.

    if (nsysdkCollectorWaitForState(NSYSDK_COLLECTOR_STATE_COLLECTING, 0) != NSYSDK_SUCCESS)
        return 1; // Profiling didn't start.

    while (nsysdkCollectorGetState() == NSYSDK_COLLECTOR_STATE_COLLECTING)
    {
        // Profiling data collection is active at this point.
        // A real plugin would be collecting and emitting data here.
        nvtxRangePushA("Workload imitation");
        std::this_thread::sleep_for(std::chrono::seconds(1));
        nvtxRangePop();
    }

    if (nsysdkCollectorGetState() != NSYSDK_COLLECTOR_STATE_FINALIZING)
        return 1; // Profiling might have been cancelled.

    // The data collection scope has been extended, emit NVTX deferred events here.
    nvtxRangePushA("Deferred events imitation");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    nvtxRangePop();

    return 0;
}
```

#### In-process shared library plugin

Nsight Systems supports plugins in the form of shared libraries loaded into the profiled application process. Such plugins are analogous to the standalone plugin type with the exception of their lifetime management: in-process plugins are loaded once and never unloaded after. If there are resources that the plugin must release, then it should use the NsysDK Collector API to track the profiler state.

Since there’s no portable analog of the `main` function in shared libraries, in-process plugins should export a function that serves as a replacement: receives user-provided arguments and starts any processing threads if needed. The signature of the initialization function is:

``` cpp
int PluginLibraryInit(int argc, const char* argv[])
```

The initialization function is called before the profiled application begins executing its `main` function. If multiple in-process plugins are enabled, they are initialized sequentially in an unspecified order. Because there’s no timeout for the initialization function execution, it should finish in a reasonable amount of time to avoid blocking the rest of profiling data collection. If an in-process plugin actively collects data rather than does some one-off modification it’ll typically launch and detach a thread in the initialization function. A plugin may return a non-zero exit code from the initialization function to indicate an error and generate a diagnostic message.

Another difference from standalone plugins is that there’s no automatic collection of NVTX events from the profiled application. If an in-process plugin generates events that should be collected, their event types need to be manually selected.

### Import NVTXT

**ImportNvtxt** is an utility which allows conversion of a [NVTXT](https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/analysis_nvtxt_file_extension.htm) file to a Nsight Systems report file (\*.nsys-rep) or to merge it with an existing report file.

Note

NvtxtImport supports custom **TimeBase** values. Only the following values are supported:

- **Manual** — timestamps are set using absolute values.

- **Relative** — timestamps are set using relative values with regards to report file which is being merged with the NVTXT file.

- **ClockMonotonicRaw** — timestamps values in the NVTXT file are considered to be gathered on the same target as the report file which is to be merged with NVTXT using `clock_gettime(CLOCK_MONOTONIC_RAW,`` ``...)` call.

- **CNTVCT** — timestamps values in the NVTXT file are considered to be gathered  
  on the same target as the report file which is to be merged with NVTXT using CNTVCT values.

You can get usage info via the help message.

Print the help message:

``` cpp
-h [ --help ]
```

Show information about the report file:

``` cpp
--cmd info -i [--input] arg
```

Create the report file from an existing NVTXT file:

``` cpp
--cmd create -n [--nvtxt] arg -o [--output] arg [-m [--mode] mode_name mode_args] [--target <Hw:Vm>] [--update_report_time]
```

Merge the NVTXT file to an existing report file:

``` cpp
--cmd merge -i [--input] arg -n [--nvtxt] arg -o [--output] arg [-m [--mode] mode_name mode_args] [--target <Hw:Vm>] [--update_report_time]
```

Modes’ descriptions:

- lerp - Insert with linear interpolation

  ``` cpp
  --mode lerp --ns_a arg --ns_b arg [--nvtxt_a arg --nvtxt_b arg]
  ```

- lin - insert with linear equation

  ``` cpp
  --mode lin  --ns_a arg --freq arg [--nvtxt_a arg]
  ```

Modes’ parameters:

- `ns_a` - a nanoseconds value

- `ns_b` - a nanoseconds value (greater than `ns_a`)

- `nvtxt_a` - an nvtxt file’s time unit value corresponding to `ns_a` nanoseconds

- `nvtxt_b` - an nvtxt file’s time unit value corresponding to `ns_b` nanoseconds

- `freq` - the nvtxt file’s timer frequency

- `--target`` ``<Hw:Vm>` - specify target id, e.g. `--target`` ``0:1`

- `--update_report_time` - prolong report’s profiling session time while  
  merging if needed. Without this option all events outside the profiling session time window will be skipped during merging.

#### Commands

**Info**

To find out report’s start and end time use **info** command.

Usage:

``` cpp
ImportNvtxt --cmd info -i [--input] arg
```

Example:

``` cpp
ImportNvtxt info Report.nsys-rep
Analysis start (ns) 83501026500000
Analysis end (ns)   83506375000000
```

**Create**

You can create a report file using existing NVTXT with **create** command.

Usage:

``` cpp
ImportNvtxt --cmd create -n [--nvtxt] arg -o [--output] arg [-m [--mode] mode_name mode_args]
```

Available modes are:

- **lerp** — insert with linear interpolation.

- **lin** — insert with linear equation.

Usage for **lerp** mode is:

``` cpp
--mode lerp --ns_a arg --ns_b arg [--nvtxt_a arg --nvtxt_b arg]
```

with:

- `ns_a` — a nanoseconds value.

- `ns_b` — a nanoseconds value (greater than `ns_a`).

- `nvtxt_a` — an nvtxt file’s time unit value corresponding to `ns_a` nanoseconds.

- `nvtxt_b` — an nvtxt file’s time unit value corresponding to `ns_b` nanoseconds.

If `nvtxt_a` and `nvtxt_b` are not specified, they are respectively set to nvtxt file’s minimum and maximum time value.

Usage for **lin** mode is:

``` cpp
--mode lin --ns_a arg --freq arg [--nvtxt_a arg]
```

with:

- `ns_a` — a nanoseconds value.

- `freq` — the nvtxt file’s timer frequency.

- `nvtxt_a` — an nvtxt file’s time unit value corresponding to `ns_a` nanoseconds.

If `nvtxt_a` is not specified, it is set to nvtxt file’s minimum time value.

**Examples:**

``` cpp
ImportNvtxt --cmd create -n Sample.nvtxt -o Report.nsys-rep
```

The output will be a new generated report file which can be opened and viewed by Nsight Systems.

**Merge**

To merge NVTXT file with an existing report file use **merge** command.

Usage:

``` cpp
ImportNvtxt --cmd merge -i [--input] arg -n [--nvtxt] arg -o [--output] arg [-m [--mode] mode_name mode_args]
```

Available modes are:

- **lerp** — insert with linear interpolation.

- **lin** — insert with linear equation.

Usage for **lerp** mode is:

``` cpp
--mode lerp --ns_a arg --ns_b arg [--nvtxt_a arg --nvtxt_b arg]
```

with:

- `ns_a` — a nanoseconds value.

- `ns_b` — a nanoseconds value (greater than `ns_a`).

- `nvtxt_a` — an nvtxt file’s time unit value corresponding to `ns_a` nanoseconds.

- `nvtxt_b` — an nvtxt file’s time unit value corresponding to `ns_b` nanoseconds.

If `nvtxt_a` and `nvtxt_b` are not specified, they are respectively set to nvtxt file’s minimum and maximum time value.

Usage for **lin** mode is:

``` cpp
--mode lin  --ns_a arg --freq arg [--nvtxt_a arg]
```

with:

- `ns_a` — a nanoseconds value.

- `freq` — the nvtxt file’s timer frequency.

- `nvtxt_a` — an nvtxt file’s time unit value corresponding to `ns_a` nanoseconds.

If `nvtxt_a` is not specified, it is set to nvtxt file’s minimum time value.

Time values in `<filename.nvtxt>` are assumed to be nanoseconds if no mode specified.

Example

``` cpp
ImportNvtxt --cmd merge -i Report.nsys-rep -n Sample.nvtxt -o NewReport.nsys-rep
```

## Reading Your Report in GUI

### Generating a New Report

Users can generate a new report by stopping a profiling session. If a profiling session has been canceled, a report will not be generated, and all collected data will be discarded.

A new `.nsys-rep` file will be created and put into the same directory as the project file (`.qdproj`).

### Opening an Existing Report

An existing `.nsys-rep` file can be opened using **File \> Open…**.

### Sharing a Report File

Report files (`.nsys-rep`) are self-contained and can be shared with other users of Nsight Systems. The only requirement is that the same or newer version of Nsight Systems is always used to open report files.

Project files (`.qdproj`) are currently not shareable, since they contain full paths to the report files.

To quickly navigate to the directory containing the report file, right click on it in the Project Explorer, and choose **Show in folder…** in the context menu.

### Report Tab

While generating a new report or loading an existing one, a new tab will be created. The most important parts of the report tab are:

- **View selector** — Allows switching between *Multi-report view* (absent for single reports), *Analysis Summary*, *Timeline View*, *Diagnostics Summary*, and *Symbol Resolution Logs* views.

  > 

- **Timeline** — This is where all charts are displayed.

- **Function table** — Located below the timeline, it displays statistical information about functions in the target application in multiple ways.

Additionally, the following controls are available:

- **Zoom slider** — Allows you to vertically zoom the charts on the timeline.

### Analysis Summary View

This view shows a summary of the profiling session. It can be used to review the various configurations used to generate this report. Depending on the features used, different subsections may be shown, but they may include

- **Profiling session information** - including information about the capture time, duration, report file, and host information.

- **Target information** - including OS and driver versions.

- **Process summary** - processes run during analysis, arguments, and CPU utilization.

- **Module summary** - modules used including name and CPU time (overall and per-process). **Note** - Module percentage in Analysis Summary page is calculated based on the cpu cycles logged in an IP sample for that module. When cpu cycles is inaccurate or absent (as is the case on x86_64 target), the module percentage is inaccurate.

- **Thread summary** - including process ID, name, and CPU utilization

- **CPU information** - information about all CPUs on the system

- **GPU information** - information about all GPUs on the system

- **Network hardware info** - information about NICs/Switches/Storage devices analyzed in the run.

- **Analysis options** - information about which Nsight Systems options were used when generating this report.

- **Nsight Systems** - information about the version used to collect and display the results.

Information from this view can be selected and copied using the mouse cursor.

### Diagnostics Summary View

This view shows important messages. Some of them were generated during the profiling session, while some were added while processing and analyzing data in the report. Messages can be one of the following types:

- Informational messages

- Warnings

- Errors

To draw attention to important diagnostics messages, a summary line is displayed on the timeline view in the top right corner:

> 

Information from this view can be selected and copied using the mouse cursor.

### Symbol Resolution Logs View

This view shows all messages related to the process of resolving symbols. It might be useful to debug issues when some of the symbol names in the symbols table of the timeline view are unresolved.

### Timeline View

The timeline view consists of two main controls: the timeline at the top, and a bottom pane that contains the events view and the function table. In some cases, when sampling of a process has not been enabled, the function table might be empty and hidden.

The bottom view selector sets the view that is displayed in the bottom pane.

> 

#### Timeline

Timeline is a versatile control that contains a tree-like **hierarchy** on the left, a *line labels* column in the center, and the corresponding *charts* on the right. The line labels column can be hidden by using the timeline options.



Contents of the hierarchy depend on the project settings used to collect the report. For example, if a certain feature has not been enabled, corresponding rows will not be shown on the timeline.

**Process Coloring**

The CPU utilization timelines are colored based on the CPU operating mode:

- User mode - Green

- Kernel mode - Red

- Other (for example system-wide processes) - Black

**CPU Activity Computation Notes**

CPU usage and thread-state ranges in Timeline are computed from the best available source:

- If CPU scheduling events are available, they are used as the primary source.

- If scheduling events are not available (for example, `--cpuctxsw=none`), and OS Runtime trace data is present, thread states are estimated from OSRT events.

**Exporting from Timeline**

To generate a timeline screenshot without opening the full GUI, use the command:

``` cpp
nsys-ui.exe --screenshot filename.nsys-rep
```

Hovering over elements in the GUI will cause a tooltip to pop open as appropriate to give additional information, such as the parameters of that function call or or the call stack. Tooltips can be copied by hovering and right clicking to bring up the `Copy`` ``Tooltip` option in the context menu:



#### Timeline Navigation

#### Zoom and Scroll

At the upper right portion of your Nsight Systems GUI, you will see this section:

> 

The slider sets the vertical size of screen rows, and the magnifying glass resets it to the original settings.

There are many ways to zoom and scroll horizontally through the timeline. Clicking on the keyboard icon seen above, opens the below dialog that explains them.

> 

#### Pinning Rows

In order to better allow users to compare rows from different sections of the timeline, Nsight Systems gives the user the ability to select rows and “pin” them in the visual range. To select a row to pin, use `Ctrl+P` or `Pin`` ``row` from the right click menu.



Once a row has been pinned, it remain at the top or bottow of the window, rather than scrolling off.



#### Timeline Correlation

The Nsight Systems GUI can correlate between calls on the CPU and GPU to help you understand the workflow.

Selecting an item will highlight that item in teal as well as:

- Any copy of that same item in other rows. This means that if there is a summary row that includes this item it will also have the appropriate section highlighted.

- All correlated items. For example, if a CUDA kernel was launched by a CPU function both are highlighted.

- All things in that thread or stream that falls into the time range since they are of part of that larger range. For example, if you clicked an NVTX range it would select all NVTX ranges and CUDA launches inside and then extend to its correlations.

The highlighting also includes lines to each event to better distinguish when highlighted events or ranges are overlapping.

In addition, Nsight Systems also provides indicators to help you find correlated items not currently on your screen, including:

- Highlights in the row headers when there is something highlighted in that row.

- Diagonal arrows in row headers if something is in a child row.

- Highlights in the timeline rule.

- Arrows in corners when something highlighted is off-screen.  You can click those and the timeline will pan or zoom to get them into view.

  

Correlation exists bidirectionally for:

- CUDA kernels, CUDA graphs, GPU memcopies, and OptiX.

- GPU memsets.

- Vulkan QueueSubmits API and CommandBuffers on GPU.

- D3D and GL just like Vulkan.

#### Timeline/Events Correlation

To display trace events in the Events View right-click a timeline row and select the `Show`` ``in`` ``Events`` ``View` command. The events of the selected row and all of its sub-rows will be displayed in the Events View. Note that the events displayed will correspond to the current zoom in the timeline, zooming in or out will reset the event pane filter.

If a timeline row has been selected for display in the Events View, then double-clicking a timeline item on that row will automatically scroll the content of the Events View to make the corresponding events view item visible and select it. If that event has tool tip information, it will be displayed in the right hand pane.

Likewise, double-clicking on a particular instance in the Events View will highlight the corresponding event in the timeline.

> 

#### Row Height

Several of the rows in the timeline use height as a way to model the percent utilization of resources. This gives the user insight into what is going on even when the timeline is zoomed all the way out.

> 

In this picture, you see that for kernel occupation there is a colored bar of variable height.

Nsight Systems calculates the average occupancy for the period of time represented by particular pixel width of screen. It then uses that average to set the top of the colored section. So, for instance, if 25% of that timeslice the kernel is active, the bar goes 25% of the distance to the top of the row.

In order to make the difference clear, if the percentage of the row height is non-zero, but would be represented by less than one vertical pixel, Nsight Systems displays it as one pixel high. The gray height represents the maximum usage in that time range.

This row height coding is used in the CPU utilization, thread and process occupancy, kernel occupancy, and memory transfer activity rows.

#### Row Percentage

In the previous image, you also see that there are percentages prefixing the stream rows in the GPU.

The percentage shown in front of the stream indicates the proportion of context running time this particular stream takes.

> ``` text
> % stream = 100.0 X streamUsage / contextUsage
> streamUsage = total amount of time this stream is active on GPU
> contextUsage = total amount of time all streams for this context are active on GPU
> ```

So “26% Stream 1” means that Stream 1 takes 26% of its context’s total running time.

> ``` text
> Total running time = sum of durations of all kernels and memory ops that
> run in this context
> ```

#### Timeline Options

We strongly recommend using the OS/Desktop defaults for size and color, but if you would like to set them for yourself, they are available using the **Tools \> Options** dialog.

The above will change the options globally for this GUI. It’s also possible to change some options for a particular open report. There is an “Options…” button near the View Selector:

> 

This button will show a dialog that allows showing/hiding the following:

- correlation arrows;

- line labels;

- CPU occupancy chart.

By default, the timeline will be based on session time. If you would like to switch to global time, click on the small arrow at the top of the leftmost column to reveal the dropdown shown below:

> 

### Events View

The Events View provides a tabular display of the trace events. The view contents can be searched and sorted.

Double-clicking an item in the Events View automatically focuses the Timeline View on the corresponding timeline item.

API calls, GPU executions, and debug markers that occurred within the boundaries of a debug marker are displayed nested to that debug marker. Multiple levels of nesting are supported.

Events view recognizes these types of debug markers:

- NVTX

- Vulkan VK_EXT_debug_marker markers, VK_EXT_debug_utils labels

- PIX events and markers

- OpenGL KHR_debug markers



You can copy and paste from the events view by highlighting rows, using **Shift** or **Ctrl** to enable multi-select. Right clicking on the selection will give you a copy option.



Pasting into text gives you a tab separated view:



Pasting into spreadsheet properly copies into rows and columns:



#### Function Table Modes

> 

The function table can work in three modes:

- **Top-Down View** — In this mode, expanding top-level functions provides information about the *callee* functions. One of the top-level functions is typically the main function of your application, or another entry point defined by the runtime libraries.

- **Bottom-Up View** — This is a reverse of the Top-Down view. On the top level, there are functions directly hit by the sampling profiler. To explore all possible call chains leading to these functions, you need to expand the subtrees of the top-level functions.

- **Flat View** — This view enumerates all functions ever observed by the profiler, even if they have never been directly hit, but just appeared somewhere on the call stack. This view typically provides a high-level overview of which parts of the code are CPU-intensive.

Each of the views helps understand particular performance issues of the application being profiled. For example:

- When trying to find specific bottleneck functions that can be optimized, the Bottom-Up view should be used. Typically, the top few functions should be examined. Expand them to understand in which contexts they are being used.

- To navigate the call tree of the application and while generally searching for algorithms and parts of the code that consume unexpectedly large amount of CPU time, the Top-Down view should be used.

- To quickly assess which parts of the application, or high level parts of an algorithm, consume significant amount of CPU time, use the Flat view.

The Top-Down and Bottom-Up views have *Self* and *Total* columns, while the Flat view has a *Flat* column. It is important to understand the meaning of each of the columns:

- Top-Down view

  - **Self** column denotes the relative amount of time spent executing instructions of this particular function.

  - **Total** column shows how much time has been spent executing this function, including all other functions called from this one. Total values of sibling rows sum up to the Total value of the parent row, or 100% for the top-level rows.

- Bottom-Up view

  - **Self** column for *top-level rows*, as in the Top-Down view, shows how much time has been spent directly in this function. Self times of all top-level rows add up to 100%.

  - **Self** column for *children rows* breaks down the value of the parent row based on the various call chains leading to that function. Self times of sibling rows add up to the value of the parent row.

- Flat view

  - **Flat** column shows how much time this function has been anywhere on the call stack. Values in this column do not add up or have other significant relationships.

Note

If low-impact functions have been filtered out, values may not add up correctly to 100%, or to the value of the parent row. This filtering can be disabled.

Contents of the symbols table is tightly related to the timeline. Users can apply and modify filters on the timeline, and they will affect which information is displayed in the symbols table:

- **Per-thread filtering** — Each thread that has sampling information associated with it has a checkbox next to it on the timeline. Only threads with selected checkboxes are represented in the symbols table.

- **Time filtering** — A time filter can be setup on the timeline by pressing the left mouse button, dragging over a region of interest on the timeline, and then choosing **Filter by selection** in the dropdown menu. In this case, only sampling information collected during the selected time range will be used to build the symbols table.

Note

If too little sampling data is being used to build the symbols table (for example, when the sampling rate is configured to be low, and a short period of time is used for time-based filtering), the numbers in the symbols table might not be representative or accurate in some cases.

#### Function Table Notes

**Last Branch Records vs. Frame Pointers**

Two of the mechanisms available for collecting backtraces are Intel Last Branch Records (LBRs) and frame pointers. LBRs are used to trace every branch instruction via a limited set of hardware registers. They can be configured to generate backtraces but have finite depth based on the CPU’s microarchitecture. LBRs are effectively free to collect but may not be as deep as you need in order to fully understand how the workload arrived a specific Instruction Pointer (IP).

Frame pointers only work when a binary is compiled with the `-fno-omit-frame-pointer` compiler switch. To determine if frame pointers are enabled on an x86_64 binary running on Linux, dump a binary’s assembly code using the `objdump`` ``-d`` ``[binary_file]` command and look for this pattern at the beginning of all functions;

``` text
push   %rbp
mov    %rsp,%rbp
```

When frame pointers are available in a binary, full stack traces will be captured. Note that libraries that are frequently used by applications and ship with the operating system, such as libc, are generated in release mode and therefore do not include frame pointers. Frequently, when a backtrace includes an address from a system library, the backtrace will fail to resolve further as the frame pointer trail goes cold due to a missing frame pointer.

A simple application was developed to show the difference. The application calls function a(), which calls b(), which calls c(), etc. Function z() calls a heavy compute function called matrix_multiply(). Almost all of the IP samples are collected while matrix_multiple is executing. The next two screen shots show one of the main differences between frame pointers and LBRs.

> 

> 

Note that the frame pointer example shows the full stack trace, while the LBR example only shows part of the stack due to the limited number of LBR registers in the CPU.

**Kernel Samples**

When an IP sample is captured while a kernel mode (i.e. operating system) function is executing, the sample will be shown with an address that starts with 0xffffffff and map to the \[kernel.kallsyms\] module.

> 

**\[vdso\]**

Samples may be collected while a CPU is executing functions in the Virtual Dynamic Shared Object. In this case, the sample will be resolved (i.e., mapped) to the \[vdso\] module. The vdso [man page](http://man7.org/linux/man-pages/man7/vdso.7.html) provides the following description of the vdso:

``` text
The “vDSO“ (virtual dynamic shared object) is a small shared library
that the kernel automatically maps into the address space of all
user-space applications.  Applications usually do not need to concern
themselves with these details as the vDSO is most commonly called by
the C library.  This way you can code in the normal way using
standard functions and the C library will take care of using any
functionality that is available via the vDSO.

Why does the vDSO exist at all?  There are some system calls the
kernel provides that user-space code ends up using frequently, to the
point that such calls can dominate overall performance. This is due
both to the frequency of the call as well as the context-switch
overhead that results from exiting user space and entering the
kernel.
```

**\[Unknown\]**

When an address can not be resolved (i.e., mapped to a module), its address within the process’ address space will be shown and its module will be marked as \[Unknown\].

#### Filter Dialog

> 

- **Collapse unresolved lines** is useful if some of the binary code does not have symbols. In this case, subtrees that consist of only unresolved symbols get collapsed in the Top-Down view, since they provide very little useful information.

- **Hide functions with CPU usage below X%** is useful for large applications, where the sampling profiler hits lots of function just a few times. To filter out the “long tail,” which is typically not important for CPU performance bottleneck analysis, this checkbox should be selected.

#### Example of Using Timeline with Function Table

Here is an example walkthrough of using the timeline and function table with Instruction Pointer (IP)/backtrace Sampling Data

**Timeline**

When a collection result is opened in the Nsight Systems GUI, there are multiple ways to view the CPU profiling data - especially the CPU IP / backtrace data.



In the timeline, yellow-orange marks can be found under each thread’s timeline that indicate the moment an IP / backtrace sample was collected on that thread (e.g., see the yellow-orange marks in the Specific Samples box above). Hovering the cursor over a mark will cause a tooltip to display the backtrace for that sample.

Below the Timeline is a drop-down list with multiple options including Events View, Top-Down View, Bottom-Up View, and Flat View. All four of these views can be used to view CPU IP / backtrace sampling data.

If the Bottom-Up View is selected, here is the sampling summary shown in the bottom half of the Timeline View screen. Notice that the summary includes the phrase “65,022 samples are used,” indicating how many samples are summarized. By default, functions that were found in less less than 0.5% of the samples are not show. Use the `filter` button to modify that setting.



When sampling data is filtered, the Sampling Summary will summarize the selected samples. Samples can be filtered on an OS thread basis, on a time basis, or both. Above, deselecting a checkbox next to a thread removes its samples from the sampling summary. Dragging the cursor over the timeline and selecting “Filter and Zoom In” chooses the samples during the time selected, as seen below. The sample summary includes the phrase “0.35% (225 samples) of data is shown due to applied filters” indicating that only 225 samples are included in the summary results.



Deselecting threads one at a time by deselecting their checkbox can be tedious. Click on the down arrow next to a thread and choose Show Only This Thread to deselect all threads except that thread.



If the Events View is selected in the Timeline View’s drop-down list, right click on a specific thread and choose Show in Events View. The samples collected while that thread executed will be shown in the Events View. Double-clicking on a specific sample in the Events view causes the timeline to show when that sample was collected; see the green boxes below. The backtrace for that sample is also shown in the Events View.



**Backtraces**

To understand the code path used to get to a specific function shown in the sampling summary, right-click on a function and select Expand.



The above shows what happens when a function’s backtraces are expanded. In this case, the PCQueuePop function was called from the CmiGetNonLocal function which was called by the CsdNextMessage function which was called by the CsdScheduleForever function. The \[Max depth\] string marks the end of the collected backtrace.



Note that, by default, backtraces with less than 0.5% of the total backtraces are hidden. This behavior can make the percentage results hard to understand. If all backtraces are shown (i.e., the filter is disabled), the results look very different and the numbers add up as expected. To disable the filter, click on the Filter… button and uncheck the **Hide functions with CPU usage below X%** checkbox.



When the filter is disabled, the backtraces are recalculated. Note that you may need to right-click on the function and select **Expand** again to get all of the backtraces to be shown.



When backtraces are collected, the whole sample (IP and backtrace) is handled as a single sample. If two samples have the exact same IP and backtrace, they are summed in the final results. If two samples have the same IP but a different backtrace, they will be shown as having the same leaf (i.e., IP) but a different backtrace. As mentioned earlier, when backtraces end, they are marked with the \[Max depth\] string (unless the backtrace can be traced back to its origin; e.g., \_\_libc_start_main) or the backtrace breaks because an IP cannot be resolved.

Above, the leaf function is PCQueuePop. In this case, there are 11 different backtraces that lead to PCQueuPop — all of them end with \[Max depth\]. For example, the dominant path is `PCQueuPop<-CmiGetNonLocal<-CsdNextmessage<-CsdScheduleForever<-[Max`` ``depth]`. This path accounts for 5.67% of all samples as shown in line 5 (red numbers). The second most dominant path is `PCQueuPop<-CmiGetNonLocal<-[Max`` ``depth]` which accounts for 0.44% of all samples as shown in line 24 (red numbers). The path `PCQueuPop<-CmiGetNonLocal<-CsdNextmessage<-CsdScheduleForever<-Sequencer::integrate(int)<-[Max`` ``depth]` accounts for 0.03% of the samples as shown in line 7 (red numbers). Adding up percentages shown in the \[Max depth\] lines (lines 5, 7, 9, 13, 15, 16, 17, 19, 21, 23, and 24) generates 7.04% which equals the percentage of samples associated with the PCQueuePop function shown in line 0 (red numbers).

### Multi-Report Timeline Views

#### Viewing Multiple Reports in Separate Panes

You have the option of looking at two or more Nsight Systems results files in separate panes. To do so, open each in a tab. Grab one of the tabs and undock:

> 

When you hover with the cursor in the middle of the GUI, you will see options for where to dock the pane:

> 

Multiple reports can be docked in the window.

> 

#### Viewing Multiple Reports in the Same Timeline

You can open several reports in a single timeline. This could be done using one of these methods:

- **File \> Open…** in the main menu, and select several report files.

- **File \> New multi-report view** in the main menu, add report files that you want to open in the Multi-report view, and click the “Apply” button.

  

Multi-report view contains simple editor that allows to add/remove some report files and will load them all on a single timeline after applying that set of reports.

> 

When reports are loaded, one can use the View Selector to open the Multi-report view again, change the set of reports, and click on “Apply” button to reload the timeline with the new set of reports.

The selected set of reports can be saved as a Multi-report view document and could be opened later to load the same set again.

#### Time Synchronization

When multiple reports are loaded into a single timeline, timestamps between them need to be adjusted, such that events that happened at the same time appear to be aligned.

Nsight Systems can automatically adjust timestamps based on **UTC time** recorded around the collection start time. This method is used by default when other more precise methods are not available. This time can be seen as `UTC`` ``time`` ``at`` ``t=0` in the *Analysis Summary* page of the report file. Refer to your OS documentation to learn how to sync the software clock using the Network Time Protocol (NTP). NTP-based time synchronization is not very precise, with the typical errors on the scale of one to tens of milliseconds.

Reports collected on the same physical machine can use synchronization based on **Timestamp Counter (TSC) values**. These are platform-specific counters, typically accessed in user space applications using the RDTSC instruction on x86_64 architecture, or by reading the CNTVCT register on Arm64. Their values converted to nanoseconds can be seen as `TSC`` ``value`` ``at`` ``t=0` in the *Analysis Summary* page of the report file. Reports synchronized using TSC values can be aligned with nanoseconds-level precision.

TSC-based time synchronization is activated automatically, when Nsight Systems detects that reports come from same target and that the same TSC value corresponds to very close UTC times. Targets are considered to be the same when either explicitly set environment variables `NSYS_HW_ID` are the same for both reports or when target hostnames are the same and `NSYS_HW_ID` is not set for either target. The difference between UTC and TSC time offsets must be below 1 second to choose TSC-based time synchronization.

To find out which synchronization method was used, navigate to the *Analysis Summary* tab of an added report and check the `Report`` ``alignment`` ``source` property of a target. Note, that the first report won’t have this parameter.

 

When loading multiple reports into a single timeline, it is always advisable to first check that time synchronization looks correct, by zooming into synchronization or communication events that are expected to be aligned.

#### Timeline Hierarchy

When reports are added to the same timeline Nsight Systems will automatically line them up by timestamps as described above. If you want Nsight Systems to also recognize matching process or hardware information, you will need to set environment variables `NSYS_SYSTEM_ID` and `NSYS_HW_ID` as shown below at the time of report collection (such as when using the “nsys profile …” command).

When loading a pair of given report files into the same timeline, they will be merged in one of the following configurations:

- **Different hardware** — is used when reports are coming from different physical machines, and no hardware resources are shared in these reports. This mode is used when neither `NSYS_HW_ID` or `NSYS_SYSTEM_ID` is set and target hostnames are different or absent, and can be additionally signalled by specifying different `NSYS_HW_ID` values.

- **Different systems, same hardware** — is used when reports are collected on different virtual machines (VMs) or containers on the same physical machine. To activate this mode, specify the same value of `NSYS_HW_ID` when collecting the reports.

- **Same system** — is used when reports are collected within the same operating system (or container) environment. In this mode a process identifier (PID) 100 will refer to the same process in both reports. To manually activate this mode, specify the same value of `NSYS_SYSTEM_ID` when collecting the reports. This mode is automatically selected when target hostnames are the same and neither `NSYS_HW_ID` or `NSYS_SYSTEM_ID` is provided.

The following diagrams demonstrate typical cases:



#### Example: MPI

A typical scenario is when a computing job is run using one of the MPI implementations. Each instance of the app can be profiled separately, resulting in multiple report files. For example:

``` cpp
# Run MPI job without the profiler:
mpirun <mpirun-options> ./myApp
# Run MPI job and profile each instance of the application:
mpirun <mpirun-options> nsys profile -o report-%p <nsys-options>./myApp
```

When each MPI rank runs on a different node, the command above works fine, since the default pairing mode (different hardware) will be used.

When all MPI ranks run the localhost only, use this command (value “A” was chosen arbitrarily, it can be any non-empty string):

``` text
NSYS_SYSTEM_ID=A mpirun <mpirun-options> nsys profile -o report-%p < nsys -options> ./myApp
```

For convenience, the MPI rank can be encoded into the report filename. For Open MPI, use the following command to create report files based on the global rank value:

``` text
mpirun <mpirun-options> nsys profile -o report-%q{OMPI_COMM_WORLD_RANK} < nsys -options> ./myApp
```

MPICH-based implementations set the environment variable `PMI_RANK` and Slurm (`srun`) provides the global MPI rank in `SLURM_PROCID`.

#### Limitations on Syncing Multiple Reports in Timeline

- Only report files collected with Nsight Systems version 2021.3 and newer are fully supported.

- Sequential reports collected in a single CLI profiling session cannot be loaded into a single timeline yet.

### Add-on Graphs - Flame Graph

The generation of Flame Graphs from Nsight Systems reports is not a built-in feature, but it is possible to create such graphs from Nsight Systems reports with the script `stackcollapse_nsys.py` located at `<nsys-install-dir>/<host-folder>/Scripts/Flamegraph`. There is also a `README.md`` ``file` at that location with additional usage details.

**Requirements:**

- `flamegraph.pl` from [Brendan Gregg’s FlameGraph github](https://github.com/brendangregg/FlameGraph),

- Perl

**Usage**

Generating flamegraph from Nsight Systems report file on Linux:

``` text
python3 stackcollapse_nsys.py report.nsys-rep | ./flamegraph.pl > result_flamegraph.svg
```

Generating flamegraph from Nsight Systems report file on Windows:

``` text
PowerShell -Command "python stackcollapse_nsys.py report.nsys-rep | perl flamegraph.pl > result_flamegraph.svg"
```

The script exports the report to SQLite, queries the CPU samples and passes them as input to flamegraph.pl.

**Parameters**

The following parameters can be passed to the script:

| Short | Long | Default | Switch Description |
|----|----|----|----|
|  | `--nsys` | Current Nsight Systems CLI installation location | Path to the Nsight Systems CLI directory (e.g., `/opt/nvidia/nsight-systems/2022.4.1/target-linux-x64`). |
| -o | `--out` | Output is written to stdout | Path to a result file containing a data suitable for `flamegraph.pl`. |
|  | `--full_function_names` | False | Use full function names with return type, arguments and expanded templates, if available. |

Note

By default, the script tries to shorten function definitions (eliminating return type, arguments and templates). In some complex cases shortening may fail and return a full function definition. To disable shortening defining `--full_function_names=False` argument can be used.

Here is an example of a Flame Graph generated from an Nsight Systems report. The program was a debug build of GROMACS, running on two ranks, each running two OpenMP threads.

> 

## Visual Studio Integration

NVIDIA Nsight Integration is a Visual Studio extension that allows you to access the power of Nsight Systems from within Visual Studio.

When Nsight Systems is installed along with NVIDIA Nsight Integration, Nsight Systems activities will appear under the NVIDIA Nsight menu in the Visual Studio menu bar. These activities launch Nsight Systems with the current project settings and executable.



Selecting the “Trace” command will launch Nsight Systems, create a new Nsight Systems project and apply settings from the current Visual Studio project:

- Target application path

- Command line parameters

- Working folder

If the “Trace” command has already been used with this Visual Studio project then Nsight Systems will load the respective Nsight Systems project and any previously captured trace sessions will be available for review using the Nsight Systems project explorer tree.

For more information about using Nsight Systems from within Visual Studio, please visit

- [NVIDIA Nsight Integration](https://docs.nvidia.com/nsight-vs-integration/index.html)

## Troubleshooting

### General Troubleshooting

**Profiling**

If the profiler behaves unexpectedly during the profiling session, or the profiling session fails to start, try the following steps:

- Close the host application.

- Restart the target device.

- Start the host application and connect to the target device.

Nsight Systems uses a settings file (`NVIDIA`` ``Nsight`` ``Systems.ini`) on the host to store information about loaded projects, report files, window layout configuration, etc. Location of the settings file is described in the **Help → About** dialog. Deleting the settings file will restore Nsight Systems to a fresh state, but all projects and reports will disappear from the Project Explorer.

**Profiling Games**

In launcher-based platforms (like Steam), if you attempt to run the game executable directly from Nsight Systems the game will detect that the launcher is missing. It will therefore launch the launcher and then self-terminate.

To avoid this, Nsight Systems on Windows automatically attaches to child processes spawned by the target process that Nsight Systems launched. Instead of setting Nsight Systems to launch the game, set Nsight Systems to launch the game platform client, and use the client GUI to launch the game.

Nsight Systems is configured to ignore the game platform client and launcher apps, and will attach to the game executable that the client launches.

An example workflow:

1.  Verify the Steam client is not running. Select the Quit command to terminate Steam if it is running.

2.  Configure Nsight Systems to launch the Steam client with a manual collection option. It is recommended you check the hotkey checkbox to begin data collection from within the game without requiring you to switching window focus.

3.  Click Start. Nsight Systems will launch the Steam client.

4.  Use Steam GUI to launch the game.

5.  When the game is running and you have reached the scene you want to profile, press the `F12` hotkey to start Nsight Systems data collection. Let the game continue running while Nsight Systems collects its profiling data (typically 10-60 seconds, or however long is relevant for you). Press `F12` again to end the collection.

**Environment Variables**

By default, Nsight Systems writes temporary files to `/tmp` directory. If you are using a system that does not allow writing to `/tmp` or where the `/tmp` directory has limited storage you can use the TMPDIR environment variable to set a different location. An example:

``` text
TMPDIR=/testdata ./bin/nsys profile -t cuda matrixMul
```

Environment variable control support for Windows target trace is not available, but there is a quick workaround:

- Create a batch file that sets the env vars and launches your application.

- Set Nsight Systems to launch the batch file as its target; i.e., set the project settings target path to the path of batch file.

- Start the trace. Nsight Systems will launch the batch file in a new cmd instance and trace any child process it launches. In fact, it will trace the whole process tree whose root is the cmd running your batch file.

**WebGL Testing**

Nsight Systems cannot profile using the default Chrome launch command. To profile WebGL please follow the following command structure:

``` text
“C:\Program Files (x86)\Google\Chrome\Application\chrome.exe”
       --inprocess-gpu --no-sandbox --disable-gpu-watchdog --use-angle=gl
       https://webglsamples.org/aquarium/aquarium.html
```

**Common Issues with QNX Targets**

- Make sure that `tracelogger` utility is available and can be run on the target.

- Make sure that `/tmp` directory is accessible and supports sub-directories.

- When switching between Nsight Systems versions, processes related to the previous version, including profiled applications forked by the daemon, must be killed before the new version is used. If you experience issues after switching between Nsight Systems versions, try rebooting the target.

#### CUDA Troubleshooting

**Flush CUDA Profile Data**

To reduce profiling overhead, the profiling tools collect and record profile information into internal buffers. These buffers are then flushed asynchronously to disk with low priority to avoid perturbing application behavior. To avoid losing profile information that has not yet been flushed, the application being profiled should make sure, before exiting, that all GPU work is done (using CUDA synchronization calls), and then call cudaProfilerStop() or cuProfilerStop(). Doing so forces buffered profile information in corresponding context(s) to be flushed.

If your CUDA application includes graphics that operate using a display or main loop, care must be taken to call cudaProfilerStop() or cuProfilerStop() before the thread executing that loop calls exit(). Failure to call one of these APIs may result in the loss of some or all of the collected profile data.

For some graphics applications like the ones use OpenGL, the application exits when the escape key is pressed. In those cases where calling the above functions before exit is not feasible, explicitly end analysis using `duration` or `nsys`` ``stop`. The profiler will force a data flush just before the timeout.

### CLI Troubleshooting

**.nsys-rep file will not load**

If you have collected a report file using the CLI and the report will not open in the GUI, check to see that your GUI version is the same or greater than the CLI version you used. If it is not, download a new version of the Nsight Systems GUI and you will be able to load and visualize your report.

This situation occurs most frequently when you update Nsight Systems using a CLI only package, such as the package available from the NVIDIA HPC SDK.

**.nsys-rep file not generated**

The CLI initially generates a .qdstrm file. The .qdstrm file is an intermediate result file, not intended for multiple imports. It needs to be processed. Usually this happens automatically. If it does not, you can use the standalone QdstrmImporter utility to generate an optimized .nsys-rep file. You can then use this file to visualize locally, to open the result on a different machine, or for sharing results with teammates.

The CLI and QdstrmImporter versions must match to convert a .qdstrm file into a .nsys-rep file. This .nsys-rep file can then be opened in the same version or more recent versions of the GUI.

To run QdstrmImporter on the host system, find the QdstrmImporter binary in the Host-x86_64 directory in your installation. QdstrmImporter is available for all host platforms. See options below.

To run QdstrmImporter on the target system, copy the Linux Host-x86_64 directory to the target Linux system or install Nsight Systems for Linux host directly on the target. The Windows or macOS host QdstrmImporter will not work on a Linux Target. See options below.

| Short | Long | Parameter | Description |
|----|----|----|----|
| -h | `--help` |  | Help message providing information about available options and their parameters. |
| -v | `--version` |  | Output QdstrmImporter version information |
| -i | `--input-file` | filename or path | Import .qdstrm file from this location. |
| -o | `--output-file` | filename or path | Provide a different file name or path for the resulting .nsys-rep file. Default is the same name and path as the .qdstrm file. |

### Launch Processes in Stopped State

In many cases, it is important to profile an application from the very beginning of its execution. When launching processes, Nsight Systems takes care of it by making sure that the profiling session is fully initialized before making the `exec()` system call on Linux.

If the process launch capabilities of Nsight Systems are not sufficient, the application should be launched manually, and the profiler should be configured to attach to the already launched process. One approach would be to call `sleep()` somewhere early in the application code, which would provide time for the user to attach to the process in Nsight Systems Embedded Platforms Edition, but there are two other more convenient mechanisms that can be used on Linux, without the need to recompile the application. (Note that the rest of this section is only applicable to Linux-based target devices.)

Both mechanisms ensure that between the time the process is created (and therefore its PID is known) and the time any of the application’s code is called, the process is stopped and waits for a signal to be delivered before continuing.

#### LD_PRELOAD

The first mechanism uses `LD_PRELOAD` environment variable. It only works with dynamically linked binaries, since static binaries do not invoke the runtime linker, and therefore are not affected by the `LD_PRELOAD` environment variable.

- For ARMv7 binaries, preload

  ``` cpp
  /opt/nvidia/nsight_systems/libLauncher32.so
  ```

- Otherwise if running from host, preload

  ``` cpp
  /opt/nvidia/nsight_systems/libLauncher64.so
  ```

- Otherwise if running from CLI, preload

  ``` cpp
  [installation_directory]/libLauncher64.so
  ```

The most common way to do that is to specify the environment variable as part of the process launch command, for example:

``` cpp
$ LD_PRELOAD=/opt/nvidia/nsight_systems/libLauncher64.so ./my-aarch64-binary --arguments
```

When loaded, this library will send itself a `SIGSTOP` signal, which is equivalent to typing `Ctrl+Z` in the terminal. The process is now a background job, and you can use standard commands like jobs, `fg` and `bg` to control them. Use `jobs`` ``-l` to see the PID of the launched process.

When attaching to a stopped process, Nsight Systems will send `SIGCONT` signal, which is equivalent to using the `bg` command.

#### Launcher

The second mechanism can be used with any binary. Use `[installation_directory]/launcher` to launch your application, for example:

``` cpp
$ /opt/nvidia/nsight_systems/launcher ./my-binary --arguments
```

The process will be launched, daemonized, and wait for `SIGUSR1` signal. After attaching to the process with Nsight Systems, the user needs to manually resume execution of the process from command line:

``` cpp
$ pkill -USR1 launcher
```

Note

Note that `pkill` will send the signal to any process with the matching name. If that is not desirable, use `kill` to send it to a specific process. The standard output and error streams are redirected to `/tmp/stdout_<PID>.txt` and `/tmp/stderr_<PID>.txt`.

The launcher mechanism is more complex and less automated than the LD_PRELOAD option, but gives more control to the user.

### GUI Troubleshooting

#### Empty or Black Pages in Analysis or Diagnostics Summary

If the **Analysis Summary** or **Diagnostics Summary** pages appear empty or black when running Nsight Systems, this may be caused by rendering issues, often related to drivers for OpenGL or Vulkan.

To resolve this, try running Nsight Systems with the following command:

``` cpp
QTWEBENGINE_CHROMIUM_FLAGS="--no-sandbox" QMLSCENE_DEVICE=softwarecontext [installation_path]/host-linux-[arch]/nsys-ui
```

#### xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin

If you encounter the following error, you may be missing the required xcb-cursor package:

``` cpp
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
```

This issue typically occurs on RHEL but may also affect other distributions. To resolve it, install the required xcb-cursor package based on your OS:

- **RHEL/CentOS/Fedora**:

  ``` cpp
  sudo dnf install -y xcb-util-cursor
  ```

- **OpenSUSE**:

  ``` cpp
  sudo dnf install -y xcb-util-cursor
  ```

- **Debian/Ubuntu**:

  ``` cpp
  sudo apt-get install -y libxcb-cursor0
  ```

#### Other Libraries Loading Errors

If opening the Nsight Systems Linux GUI fails with one of the following errors, you may be missing some required libraries:

``` cpp
This application failed to start because it could not find or load the Qt platform plugin "xcb" in "". Available platform plugins are: xcb. Reinstalling the application may fix this problem.
```

or

``` cpp
error while loading shared libraries: [library_name]: cannot open shared object file: No such file or directory
```

##### Ubuntu and CentOS With Root Privileges

- Launch the following command, which will install all the required libraries in system directories:

  ``` cpp
  [installation_path]/host-linux-[arch]/Scripts/DependenciesInstaller/install-dependencies.sh
  ```

- Launch the Linux GUI as usual.

##### Ubuntu and CentOS Without Root Privileges

- Choose the directory where dependencies will be installed (`dependencies_path`). This directory should be writeable for the current user.

- Launch the following command (if it has already been run, move to the next step), which will install all the required libraries in `[dependencies_path]`:

  ``` cpp
  [installation_path]/host-linux-[arch]/Scripts/DependenciesInstaller/install-dependencies-without-root.sh [dependencies_path]
  ```

- Further, use the following command to launch the Linux GUI:

  ``` cpp
  source [installation_path]/host-linux-[arch]/Scripts/DependenciesInstaller/setup-dependencies-environment.sh [dependencies_path] && [installation_path]/host-linux-[arch]/nsys-ui
  ```

##### Other platforms, or if the previous steps did not help

Launch Nsight Systems using the following command line to determine which libraries are missing and install them.

``` cpp
$ QT_DEBUG_PLUGINS=1 [installation_path]/host-linux-[arch]/nsys-ui
```

If the workload does not run when launched via Nsight Systems or the timeline is empty, check the stderr.log and stdout.log (click on drop-down menu showing **Timeline View** and click on **Files**) to see the errors encountered by the app.

> 

### Symbol Resolution

If stack trace information is missing symbols and you have a symbol file, you can manually re-resolve using the ResolveSymbols utility. This can be done by right-clicking the report file in the Project Explorer window and selecting “Resolve Symbols…”.

Alternatively, you can find the utility as a separate executable in the `[installation_path]\Host` directory. This utility works with ELF format files, with Linux Debuginfod cache directories and Linux/QNX symbol servers, with Windows PDB directories and symbol servers, or with files where each line is in the format `<start><length><name>`.

[TABLE]

#### Broken Backtraces on Tegra

In Nsight Systems Embedded Platforms Edition, in the symbols table there is a special entry called **Broken backtraces**. This entry is used to denote the point in the call chain where the unwinding algorithms used by Nsight Systems could not determine what is the next (caller) function.

Broken backtraces happen because there is no information related to the current function that the unwinding algorithms can use. In the Top-Down view, these functions are immediate children of the Broken backtraces row.

One can eliminate broken backtraces by modifying the build system to provide at least one kind of unwind information. The types of unwind information, used by the algorithms in Nsight Systems, include the following:

For ARMv7 binaries:

- DWARF information in ELF sections: `.debug_frame`, `.zdebug_frame`, `.eh_frame`, `.eh_frame_hdr`. This information is the most precise. `.zdebug_frame` is a compressed version of `.debug_frame`, so at most one of them is typically present. `.eh_frame_hdr` is a companion section for `.eh_frame` and might be absent.

  Compiler flag: `-g`.

- Exception handling information in EHABI format provided in `.ARM.exidx` and `.ARM.extab` ELF sections. `.ARM.extab` might be absent if all information is compact enough to be encoded into `.ARM.exidx`.

  Compiler flag: `-funwind-tables`.

- Frame pointers (built into the `.text` section).

  Compiler flag: `-fno-omit-frame-pointer`.

For Aarch64 binaries:

- DWARF information in ELF sections: `.debug_frame`, `.zdebug_frame`, `.eh_frame`, `.eh_frame_hdr`. See additional comments above.

  Compiler flag: `-g`.

- Frame pointers (built into the `.text` section).

  Compiler flag: `-fno-omit-frame-pointer`.

The following ELF sections should be considered empty if they have size of 4 bytes: `.debug_frame`, `.eh_frame`, `.ARM.exidx`. In this case, these sections only contain termination records and no useful information.

For GCC, use the following compiler invocation to see which compiler flags are enabled in your toolchain by default (for example, to check if `-funwind-tables` is enabled by default):

``` cpp
$ gcc -Q --help=common
```

For GCC and Clang, add `-###` to the compiler invocation command to see which compiler flags are actually being used.

Since EHABI and DWARF information is compiled on per-unit basis (every `.cpp` or `.c` file, as well as every static library, can be built with or without this information), presence of the ELF sections does not guarantee that every function has necessary unwind information.

Frame pointers are required by the Aarch64 Procedure Call Standard. Adding frame pointers slows down execution time, but in most cases the difference is negligible.

#### Debug Versions of ELF Files

Often, after a binary is built, especially if it is built with debug information (`-g` compiler flag), it gets stripped before deploying or installing. In this case, ELF sections that contain useful information, such as non-export function names or unwind information, can get stripped as well.

One solution is to deploy or install the original unstripped library instead of the stripped one, but in many cases this would be inconvenient. Nsight Systems can use missing information from alternative locations.

For target devices with Ubuntu, see [Debug Symbol Packages](https://wiki.ubuntu.com/Debug_Symbol_packages). These packages typically install debug ELF files with `/usr/lib/debug` prefix. Nsight Systems can find debug libraries there, and if it matches the original library (e.g., the built-in `BuildID` is the same), it will be picked up and used to provide symbol names and unwind information.

Many packages have debug companions in the same repository and can be directly installed with APT (`apt-get`). Look for packages with the `-dbg` suffix. For other packages, refer to the [Debug Symbol Packages](https://wiki.ubuntu.com/Debug_Symbol_packages) wiki page on how to add the debs package repository. After setting up the repository and running apt-get update, look for packages with `-dbgsym` suffix.

To verify that a debug version of a library has been picked up and downloaded from the target device, look in the **Module Summary** section of **Analysis Summary**:

> 

### Logging

To enable logging on the host, refer to this config file:

``` cpp
host-linux-x64/nvlog.config.template
```

When reporting any bugs please include the build version number as described in the **Help → About** dialog. If possible, attach log files and report (`.nsys-rep`) files, as they already contain necessary version information.

#### Verbose Remote Logging on Linux Targets

Verbose logging is available when connecting to a Linux-based device from the GUI on the host. This extra debug information is not available when launching via the command line. Nsight Systems installs its executable and library files into the following directory:

``` cpp
/opt/nvidia/nsight_systems/
```

To enable verbose logging on the target device, when launched from the host, follow these steps:

1.  Close the host application.

2.  Restart the target device.

3.  Place `nvlog.config` from host directory to the `/opt/nvidia/nsight_systems` directory on target.

4.  From SSH console, launch the following command:

    ``` cpp
    sudo /opt/nvidia/nsight_systems/nsys --daemon --debug
    ```

5.  Start the host application and connect to the target device.

Logs on the target devices are collected into this file (if enabled):

``` cpp
nsys.log
```

in the directory where `nsys` command was launched.

Please note that in some cases, debug logging can significantly slow down the profiler.

#### Verbose CLI Logging on Linux Targets

To enable verbose logging of the Nsight Systems CLI and the target application’s injection behavior:

1.  In the target-linux-x64 directory, rename the nvlog.config.template file to nvlog.config.

2.  Inside that file, change the line:

    ``` cpp
    $ nsys-ui.log
    ```

    to:

    ``` cpp
    $ nsys-agent.log
    ```

3.  Run a collection, either explicitly giving the path to the config file or with the config file in the same directory with the application. The `target-linux-x64` directory will then include a file named `nsys-agent.log`.

``` text
nsys profile --trace=osrt \
   --env-var=NVLOG_CONFIG_FILE="<install-dir>/target-linux-x64/nvlog.config" \
   sleep 1
```

Note

In some cases, debug logging can significantly slow down the profiler.

#### Verbose Logging on Windows Targets

Verbose logging is available when connecting to a Windows-based device from the GUI on the host. Nsight Systems installs its executable and library files into the following directory by default:

``` text
C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.3
```

To enable verbose logging on the target device, when launched from the host, follow these steps:

1.  Close the host application.

2.  Terminate the `nsys` process.

3.  Place `nvlog.config` from host directory next to Nsight Systems Windows agent on the target device.

    - Local Windows target:

      ``` text
      C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.3\target-windows-x64
      ```

    - Remote Windows target:

      ``` text
      C:\Users\<user name>\AppData\Local\Temp\nvidia\nsight_systems
      ```

4.  Start the host application and connect to the target device.

Logs on the target devices are collected into this file (if enabled):

``` text
nsight-sys.log
```

in the same directory as Nsight Systems Windows agent.

Note

In some cases, debug logging can significantly slow down the profiler.

## Other Resources

Looking for information to help you use Nsight Systems the most effectively? Here are some more resources you might want to review:

### Training Seminars

NVIDIA Deep Learning Institute Training - Self-Paced Online Course - [Nsight Analysis System: Build Custom Python Analysis Scripts](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-13+V1)

NVIDIA Deep Learning Institute Training - Self-Paced Online Course - [Optimizing CUDA Machine Learning Codes With Nsight Profiling Tools](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-03+V2)

CUDA Developer Tools YouTube Channel - [Intro to NVIDIA Nsight Systems](https://www.youtube.com/watch?v=dUDGO66IadU)

2018 NCSA Blue Waters Webinar - Video Only - [Introduction to NVIDIA Nsight Systems](https://www.youtube.com/watch?v=WA8C48FJi3c)

### Blog Posts

**NVIDIA developer blogs**

These are longer form, technical pieces written by tool and domain experts.

- 2021 : [Optimizing DX12 Resource Uploads to the GPU Using CPU-Visible VRAM](https://developer.nvidia.com/blog/optimizing-dx12-resource-uploads-to-the-gpu-using-cpu-visible-vram/)

- 2020 : [Understanding the Visualization of Overhead and Latency in Nsight Systems](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)

- 2019 : [Migrating to NVIDIA Nsight Tools from NVVP and nvprof](https://devblogs.nvidia.com/migrating-nvidia-nsight-tools-nvvp-nvprof/)

- 2019 : [Transitioning to Nsight Systems from NVIDIA Visual Profiler / nvprof](https://devblogs.nvidia.com/transitioning-nsight-systems-nvidia-visual-profiler-nvprof/)

- 2019 : [NVIDIA Nsight Systems Add Vulkan Support](https://devblogs.nvidia.com/nvidia-nsight-systems-adds-vulkan-support/)

**External Blog Posts**

- 2025 : Practical ML, [Speed Up PyTorch Training by 3x with NVIDIA Nsight and PyTorch 2.0 Tricks](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)

### Feature Videos

Short videos, only a minute or two, to introduce new features.

- [OpenMP Trace Feature Spotlight](https://youtu.be/ZeuM2k_hrq0)

- [Command Line Sessions Video Spotlight](https://youtu.be/r2ewwd4d0vc)

- [Direct3D11 Feature Spotlight](https://youtu.be/DUhzjyBr-wg)

- [Vulkan Trace](https://youtu.be/witzRF-wu8M)

- [Statistics Driven Profiling](https://www.youtube.com/watch?v=fyhPFTF75tk)

### Conference Presentations

- GTC 2025 - [Optimizing Multi-Language Scientific Simulations: A Grace Superchip Case Study](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72537/)

- GTC 2024 - [Achieving Higher Performance From Your Data Center and Cloud Application](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62388/)

- Jetson Edge AI Developer Days 2023 - [Getting the Most Out of Your Jetson Orin Using NVIDIA Nsight Developer Tools](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-se52434/)

- GTC 2023 - [Overcoming Pre- and Post-Processing Bottlenecks in AI-Based Imaging and Computer Vision Pipelines with CV-CUDA](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51182/)

- GTC 2023 - [Optimizing at Scale: Investigating Hidden Bottlenecks for Multi-Node Workloads](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51421/)

- GTC 2023 - [Optimize Multi-Node System Workloads With NVIDIA Nsight Systems](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-d52660/)

- GTC 2023 - [Ray-Tracing Development using NVIDIA Nsight Graphics and NVIDIA Nsight Systems](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-dlit51580/)

- GTC 2022 - [Killing Cloud Monsters Has Never Been Smoother](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41518/)

- GTC 2022 - [Optimizing Communication with Nsight Systems Network Profiling](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41500/)

- GTC 2022 - [Optimizing Vulkan 1.3 Applications with Nsight Graphics & Nsight Systems](https://www.nvidia.com/en-us/on-demand/session/other2021-vlk2201/)

- GTC 2021 - [Tuning GPU Network and Memory Usage in Apache Spark](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31566/)

- GTC 2020 - [Rebalancing the Load: Profile-Guided Optimization of the NAMD Molecular Dynamics Program for Modern GPUs using Nsight Systems](https://developer.nvidia.com/gtc/2020/video/s21547)

- GTC 2020 - [Scaling the Transformer Model Implementation in PyTorch Across Multiple Nodes](https://developer.nvidia.com/gtc/2020/video/s21351)

- GTC 2019 - [Using Nsight Tools to Optimize the NAMD Molecular Dynamics Simulation Program](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s9503-using+nsight+tools+to+optimize+the+namd+molecular+dynamics+simulation+program)

- GTC 2019 - [Optimizing Facebook AI Workloads for NVIDIA GPUs](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s9866-optimizing+facebook+ai+workloads+for+nvidia+gpus)

- GTC 2018 - [Optimizing HPC Simulation and Visualization Codes Using NVIDIA Nsight Systems](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s8718-optimizing+hpc+simulation+and+visualization+codes+using+the+nvidia+nsight+systems)

- GTC 2018 - Israel - [Boost DNN Training Performance using NVIDIA Tools](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=sil8105-boost+dnn+training+performance+using+nvidia+tools)

- Siggraph 2018 - [Taming the Beast; Using NVIDIA Tools to Unlock Hidden GPU Performance](http://on-demand.gputechconf.com/siggraph/2018/video/sig1805-aurelio-reis-yaki-tebeka-taming-the-beast.html)

### For More Support

To file a bug report or to ask a question on the Nsight Systems forums, you will need to register with the NVIDIA Developer Program. See the [FAQ](https://www.nvidia.com/en-us/account/faq/). You do not need to register to read the forums.

After that, you can access Nsight Systems [Forums](https://forums.developer.nvidia.com/c/development-tools/nsight-systems/116) and the [NVIDIA Bug Tracking System](https://developer.nvidia.com/nvidia_bug/add).

To submit feedback directly from the GUI, go to **Help-\>Send Feedback** and fill out the form. Enter your email address if you would like to hear back from the Nsight Systems team.

> 


---


<!-- source: https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html -->
# Post-Collection Analysis Guide

NVIDIA Nsight Systems Analysis guide.

## Overview

Once you have profiled using Nsight Systems there are many options for analyzing the collected data, as well as to output it in various formats. These options are available from the CLI or the GUI.

## Statistical Analysis

### Statistical Reports Shipped With Nsight Systems

The Nsight Systems development team created and maintains a set of report scripts for some of the commonly requested statistical reports. These scripts will be updated to adapt to any changes in SQLite schema or internal data structures.

These scripts are located in the Nsight Systems package in the Target-\<architecture\>/reports directory. The following standard reports are available:

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Note

All time values given in nanoseconds by default. If you wish to output the results using a different time unit, use the `--timeunit` option when running the recipe.

#### cuda_api_gpu_sum\[:nvtx-name\]\[:base\|:mangled\] – CUDA Summary (API/Kernels/MemOps)

Arguments

- nvtx-name : Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will cause summary to be over the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this kernel

- Instances : Number of executions of this kernel

- Avg : Average execution time of this kernel

- Med : Median execution time of this kernel

- Min : Smallest execution time of this kernel

- Max : Largest execution time of this kernel

- StdDev : Standard deviation of execution time of this kernel

- Category : Category of the operation

- Operation : Name of the kernel

This report provides a summary of CUDA API calls, kernels and memory operations, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that API call’s, kernel’s, or memory operation’s percent of the execution time of the APIs, kernels and memory operations listed, and not a percentage of the application wall or CPU execution time.

This report combines data from the `cuda_api_sum`, `cuda_gpu_kern_sum`, and `cuda_gpu_mem_size_sum` reports. It is very similar to profile section of `nvprof`` ``--dependency-analysis`.

#### cuda_api_sum – CUDA API Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this function

- Num Calls : Number of calls to this function

- Avg : Average execution time of this function

- Med : Median execution time of this function

- Min : Smallest execution time of this function

- Max : Largest execution time of this function

- StdDev : Standard deviation of the time of this function

- Name : Name of the function

This report provides a summary of CUDA API functions and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that function’s percent of the execution time of the functions listed, and not a percentage of the application wall or CPU execution time.

#### cuda_api_trace – CUDA API Trace

Arguments - None

Output: All time values default to nanoseconds

- Start : Timestamp when API call was made

- Duration : Length of API calls

- Name : API function name

- Result : Return value of API call

- CorrID : Correlation used to map to other CUDA calls

- Pid : Process ID that made the call

- Tid : Thread ID that made the call

- T-Pri : Run priority of call thread

- Thread Name : Name of thread that called API function

This report provides a trace record of CUDA API function calls and their execution times.

#### cuda_gpu_kern_gb_sum\[:nvtx-name\]\[:base\|:mangled\] – CUDA GPU Kernel/Grid/Block Summary

Arguments

- nvtx-name - Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will cause summary to be over the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this kernel

- Instances : Number of calls to this kernel

- Avg : Average execution time of this kernel

- Med : Median execution time of this kernel

- Min : Smallest execution time of this kernel

- Max : Largest execution time of this kernel

- StdDev : Standard deviation of the time of this kernel

- GridXYZ : Grid dimensions for kernel launch call

- BlockXYZ : Block dimensions for kernel launch call

- Name : Name of the kernel

This report provides a summary of CUDA kernels and their execution times. Kernels are sorted by grid dimensions, block dimensions, and kernel name. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that kernel’s percent of the execution time of the kernels listed, and not a percentage of the application wall or CPU execution time.

#### cuda_gpu_kern_sum\[:nvtx-name\]\[:base\|:mangled\] – CUDA GPU Kernel Summary

Note

In recent versions of Nsight Systems, this report was expanded to include and sort by CUDA grid and block dimensions. This change was made to accommodate developers doing a certain type of optimization work. Unfortunately, this change caused an unexpected burden for developers doing a different type of optimization work. In order to service both use-cases, this report has been returned to the original form, without grid or block information. A new report, called `cuda_gpu_kern_gb_sum`, has been created that retains the grid and block information.

Arguments

- nvtx-name - Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will cause summary to be over the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this kernel

- Instances : Number of calls to this kernel

- Avg : Average execution time of this kernel

- Med : Median execution time of this kernel

- Min : Smallest execution time of this kernel

- Max : Largest execution time of this kernel

- StdDev : Standard deviation of the time of this kernel

- Name : Name of the kernel

This report provides a summary of CUDA kernels and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that kernel’s percent of the execution time of the kernels listed, and not a percentage of the application wall or CPU execution time.

#### cuda_gpu_mem_size_sum – CUDA GPU MemOps Summary (by Size)

Arguments - None

Output:

- Total : Total memory utilized by this operation

- Count : Number of executions of this operation

- Avg : Average memory size of this operation

- Med : Median memory size of this operation

- Min : Smallest memory size of this operation

- Max : Largest memory size of this operation

- StdDev : Standard deviation of the memory size of this operation

- Operation : Name of the operation

This report provides a summary of GPU memory operations and the amount of memory they utilize.

#### cuda_gpu_mem_time_sum – CUDA GPU MemOps Summary (by Time)

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this operation

- Count : Number of operations to this type

- Avg : Average execution time of this operation

- Med : Median execution time of this operation

- Min : Smallest execution time of this operation

- Max : Largest execution time of this operation

- StdDev : Standard deviation of execution time of this operation

- Operation : Name of the memory operation

This report provides a summary of GPU memory operations and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that operation’s percent of the execution time of the operations listed, and not a percentage of the application wall or CPU execution time.

#### cuda_gpu_sum\[:nvtx-name\]\[:base\|:mangled\] – CUDA GPU Summary (Kernels/MemOps)

Arguments

- nvtx-name - Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will cause summary to be over the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this kernel

- Instances : Number of executions of this kernel

- Avg : Average execution time of this kernel

- Med : Median execution time of this kernel

- Min : Smallest execution time of this kernel

- Max : Largest execution time of this kernel

- StdDev : Standard deviation of execution time of this kernel

- Category : Category of the operation

- Operation : Name of the kernel

This report provides a summary of CUDA kernels and memory operations, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that kernel’s or memory operation’s percent of the execution time of the kernels and memory operations listed, and not a percentage of the application wall or CPU execution time.

This report combines data from the `cuda_gpu_kern_sum` and `cuda_gpu_mem_time_sum` reports. This report is very similar to output of the command `nvprof`` ``--print-gpu-summary`.

#### cuda_gpu_trace\[:nvtx-name\]\[:base\|:mangled\] – CUDA GPU Trace

Arguments

- nvtx-name - Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will display the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will display the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- Start : Timestamp of start time

- Duration : Length of event

- CorrId : Correlation ID

- GrdX, GrdY, GrdZ : Grid values

- BlkX, BlkY, BlkZ : Block values

- Reg/Trd : Registers per thread

- StcSMem : Size of Static Shared Memory

- DymSMem : Size of Dynamic Shared Memory

- Bytes : Size of memory operation

- Throughput : Memory throughput

- SrcMemKd : Memcpy source memory kind or memset memory kind

- DstMemKd : Memcpy destination memory kind

- Device : GPU device name and ID

- Ctx : Context ID

- GreenCtx: Green context ID

- Strm : Stream ID

- Name : Trace event name

This report displays a trace of CUDA kernels and memory operations. Items are sorted by start time.

#### cuda_kern_exec_sum\[:nvtx-name\]\[:base\|:mangled\] – CUDA Kernel Launch & Exec Time Summary

Arguments

- nvtx-name - Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will cause summary to be over the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- PID : Process ID that made kernel launch call

- TID : Thread ID that made kernel launch call

- DevId : CUDA Device ID that executed kernel (which GPU)

- Count : Number of kernel records

- QCount : Number of kernel records with positive queue time

Average, Median, Minimum, Maximum, and Standard Deviation for:

- TAvg, TMed, TMin, TMax, TStdDev : Total time

- AAvg, AMed, AMin, AMax, AStdDev : API time

- QAvg, QMed, QMin, QMax, QStdDev : Queue time

- KAvg, KMed, KMin, KMax, KStdDev : Kernel time

- API Name : Name of CUDA API call used to launch kernel

- Kernel Name : Name of CUDA Kernel

This report provides a summary of the launch and execution times of CUDA kernels. The launch and execution is broken down into three phases: “API time,” the execution time of the CUDA API call on the CPU used to launch the kernel; “Queue time,” the time between the launch call and the kernel execution; and “Kernel time,” the kernel execution time on the GPU. The “total time” is not a just sum of the other times, as the phases sometimes overlap. Rather, the total time runs from the start of the API call to end of the API call or the end of the kernel execution, whichever is later.

The reported queue time is measured from the end of the API call to the start of the kernel execution. The actual queue time is slightly longer, as the kernel is enqueue somewhere in the middle of the API call, and not in the final nanosecond of function execution. Due to this delay, it is possible for kernel execution to start before the CUDA launch call returns. In these cases, no queue time will be reported. Only kernel launches with positive queue times are included in the queue average, minimum, maximum, and standard deviation calculations. The “QCount” column indicates how many launches had positive queue times (and how many launches were involved in calculating the queue time statistics). Subtracting “QCount” from “Count” will indicate how many kernels had no queue time.

Be aware that having a queue time is not inherently bad. Queue times indicate that the GPU was busy running other tasks when the new kernel was scheduled for launch. If every kernel launch is immediate, without any queue time, that \_may\_ indicate an idle GPU with poor utilization. In terms of performance optimization, it should not necessarily be a goal to eliminate queue time.

#### cuda_kern_exec_trace\[:nvtx-name\]\[:base\|:mangled\] – CUDA Kernel Launch & Exec Time Trace

Arguments

- nvtx-name - Optional argument, if given, will prefix the kernel name with the name of the innermost enclosing NVTX range.

- base - Optional argument, if given, will cause summary to be over the base name of the kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note: the ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- API Start : Start timestamp of CUDA API launch call

- API Dur : Duration of CUDA API launch call

- Queue Start : Start timestamp of queue wait time, if it exists

- Queue Dur : Duration of queue wait time, if it exists

- Kernel Start : Start timestamp of CUDA kernel

- Kernel Dur : Duration of CUDA kernel

- Total Dur : Duration from API start to kernel end

- PID : Process ID that made kernel launch call

- TID : Thread ID that made kernel launch call

- DevId : CUDA Device ID that executed kernel (which GPU)

- API Function : Name of CUDA API call used to launch kernel

- GridXYZ : Grid dimensions for kernel launch call

- BlockXYZ : Block dimensions for kernel launch call

- Kernel Name : Name of CUDA Kernel

This report provides a trace of the launch and execution time of each CUDA kernel. The launch and execution is broken down into three phases: “API time,” the execution time of the CUDA API call on the CPU used to launch the kernel; “Queue time,” the time between the launch call and the kernel execution; and “Kernel time,” the kernel execution time on the GPU. The “total time” is not a just sum of the other times, as the phases sometimes overlap. Rather, the total time runs from the start of the API call to end of the API call or the end of the kernel execution, whichever is later.

The reported queue time is measured from the end of the API call to the start of the kernel execution. The actual queue time is slightly longer, as the kernel is enqueue somewhere in the middle of the API call, and not in the final nanosecond of function execution. Due to this delay, it is possible for kernel execution to start before the CUDA launch call returns. In these cases, no queue times will be reported.

Be aware that having a queue time is not inherently bad. Queue times indicate that the GPU was busy running other tasks when the new kernel was scheduled for launch. If every kernel launch is immediate, without any queue time, that \_may\_ indicate an idle GPU with poor utilization. In terms of performance optimization, it should not necessarily be a goal to eliminate queue time.

#### dx11_pix_sum – DX11 PIX Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this rage

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of D3D11 PIX CPU debug markers, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### dx12_gpu_marker_sum – DX12 GPU Command List PIX Ranges Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of DX12 PIX GPU command list debug markers, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### dx12_pix_sum – DX12 PIX Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of D3D12 PIX CPU debug markers, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### mpi_event_sum – MPI Event Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this event

- Instances : Number of instances of this event

- Avg : Average execution time of this event

- Med : Median execution time of this event

- Min : Smallest execution time of this event

- Max : Largest execution time of this event

- StdDev : Standard deviation of execution time of this event

- Source: Original source class of event data

- Name : Name of MPI event

This report provides a summary of all recorded MPI events. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that event’s percent of the total execution time of the listed events, and not a percentage of the application wall or CPU execution time.

#### mpi_event_trace – MPI Event Trace

Arguments - None

Output: All time values default to nanoseconds

- Start : Start timestamp of event

- End : End timestamp of event

- Duration : Duration of event

- Event : Name of event type

- Pid : Process Id that generated the event

- Tid : Thread Id that generated the event

- Tag : MPI message tag

- Rank : MPI Rank that generated event

- PeerRank : Other MPI rank of send or receive type events

- RootRank : Root MPI rank for broadcast type events

- Size : Size of message for uni-directional operations (send & recv)

- CollSendSize : Size of sent message for collective operations

- CollRecvSize : Size of received message for collective operations

This report provides a trace record of all recorded MPI events.

Note that MPI_Sendrecv events with different rank, tag, or size values are broken up into two separate report rows, one reporting the send, and one reporting the receive. If only one row exists, the rank, tag, and size can assumed to be the same.

#### mpi_msg_size_sum – MPI Message Size Summary

Arguments - None

Output: Message size values are in bytes

- Total Message Volume : Aggregated message size from all instances of this API function

- Instances : Number of instances of this API function

- Avg : Average message size of this API function

- Med : Median message size of this API function

- Min : Smallest message size of this API function

- Max : Largest message size of this API function

- StdDev : Standard deviation of message size for this API function

- Source : Message source (p2p, coll_send, coll_recv)

- Name : Name of the MPI API function

This report provides a message size summary of all collective and point-to-point MPI calls.

Note that for MPI collectives the report presents the sent message with Source equal to `coll_send` and the received message with Source equal to `coll_recv`.

#### network_congestion\[:ticks_threshold=\<ticks_per_ms\>\] – Network Devices Congestion

Arguments

- ticks_threshold=\<ticks_per_ms\> - Threshold in ticks/ms above which we report congestion. Default is 10000.

Output: All time values default to nanoseconds

- Start : Start timestamp of congestion interval

- End : End timestamp of congestion interval

- Duration : Duration of congestion interval

- Send wait rate: Rate of congestion during the interval

- GUID : The device GUID

- Name : The device name

This report displays congestion events with a high send wait rate. By default, only events with a send wait rate above 10000 ticks/ms are shown, but a custom threshold value can be set.

Each event defines a period of time when the device experienced some level of congestion. The level of congestion is defined by the send wait rate, given in time ticks per millisecond (ticks/ms). The specific duration of a tick is device specific, but can be assumed to be nanoseconds in scale. Congestion is measured by counting the number of ticks during which the port had data to transmit, but no data was sent because of insufficient credits or because of lack of arbitration. The presented value of send wait rate is the amount of ticks counted during an event, normalized over the event’s duration. Higher send wait rate values indicate more congestion.

Because the specific duration of a tick is device dependent, analysis should focus on the relative send wait rates of events generated by the same device. Comparing absolute send wait rates across devices is only meaningful if the time tick duration is known to be similar.

For IB Switch metrics, we do not present the device name, only the GUID.

#### nvtx_gpu_proj_sum – NVTX GPU Projection Summary

Arguments - None

Output: All time values default to nanoseconds

- Range : Name of the NVTX range

- Style : Range style; Start/End or Push/Pop

- Total Proj Time: Total projected time used by all instances of this range name

- Total Range Time: Total original NVTX range time used by all instances of this range name

- Range Instances : Number of instances of this range

- Proj Avg : Average projected time for this range

- Proj Med : Median projected time for this range

- Proj Min : Minimum projected time for this range

- Proj Max : Maximum projected time for this range

- Proj StdDev : Standard deviation of projected times for this range

- Total GPU Ops : Total number of GPU ops

- Avg GPU Ops : Average number of GPU ops

- Avg Range Lvl : Average range stack depth

- Avg Num Child : Average number of children ranges

This report provides a summary of NVTX time ranges projected from the CPU to the GPU. Each NVTX range contains one or more GPU operations. A GPU operation is considered to be “contained” by the NVTX range if the CUDA API call used to launch the operation is within the NVTX range. Only ranges that start and end on the same thread are taken into account.

The projected range will have the start timestamp of the start of the first enclosed GPU operation and the end timestamp of the end of the last enclosed GPU operation. This report then summarizes all the range instances by name and style. Note that in cases when one NVTX range might enclose another, the time of the child(ren) range(s) is not subtracted from the parent range. This is because the projected times may not strictly overlap like the original NVTX range times do. As such, the total projected time of all ranges might exceed the total sampling duration.

#### nvtx_gpu_proj_trace – NVTX GPU Projection Trace

Arguments - None

Output: All time values default to nanoseconds

- Name : Name of the NVTX range

- Projected Start : Projected range start timestamp

- Projected Duration : Projected range duration

- Orig Start : Original NVTX range start timestamp

- Orig Duration : Original NVTX range duration

- Style : Range style; Start/End or Push/Pop

- PID : Process ID

- TID : Thread ID

- NumGPUOps : Number of enclosed GPU operations

- Lvl : Stack level, starts at 0

- NumChild : Number of children ranges

- RangeId : Arbitrary ID for range

- ParentId : Range ID of the enclosing range

- RangeStack : Range IDs that make up the push/pop stack

This report provides a trace of NVTX time ranges projected from the CPU onto the GPU. Each NVTX range contains one or more GPU operations. A GPU operation is considered to be “contained” by an NVTX range if the CUDA API call used to launch the operation is within the NVTX range. Only ranges that start and end on the same thread are taken into account.

The projected range will have the start timestamp of the first enclosed GPU operation and the end timestamp of the last enclosed GPU operation, as well as the stack state and relationship to other NVTX ranges.

#### nvtx_kern_sum\[:base\|:mangled\] – NVTX Range Kernel Summary

Arguments

- base - Optional argument, if given, will cause summary to be over the base name of the CUDA kernel, rather than the templated name.

- mangled - Optional argument, if given, will cause summary to be over the raw mangled name of the kernel, rather than the templated name.

Note

The ability to display mangled names is a recent addition to the report file format, and requires that the profile data be captured with a recent version of Nsight Systems. Re-exporting an existing report file is not sufficient. If the raw, mangled kernel name data is not available, the default demangled names will be used.

Output: All time values default to nanoseconds

- NVTX Range : Name of the range

- Style : Range style; Start/End or Push/Pop

- PID : Process ID for this set of ranges and kernels

- TID : Thread ID for this set of ranges and kernels

- NVTX Inst : Number of NVTX range instances

- Kern Inst : Number of CUDA kernel instances

- Total Time : Total time used by all kernel instances of this range

- Avg : Average execution time of the kernel

- Med : Median execution time of the kernel

- Min : Smallest execution time of the kernel

- Max : Largest execution time of the kernel

- StdDev : Standard deviation of the execution time of the kernel

- Kernel Name : Name of the kernel

This report provides a summary of CUDA kernels, grouped by NVTX ranges. To compute this summary, each kernel is matched to one or more containing NVTX range in the same process and thread ID. A kernel is considered to be “contained” by an NVTX range if the CUDA API call used to launch the kernel is within the NVTX range. The actual execution of the kernel may last longer than the NVTX range. A specific kernel instance may be associated with more than one NVTX range if the ranges overlap. For example, if a kernel is launched inside a stack of push/pop ranges, the kernel is considered to be “contained” by all of the ranges on the stack, not just the deepest range. This becomes very confusing if NVTX ranges appear inside other NVTX ranges of the same name.

Once each kernel is associated to one or more NVTX range(s), the list of ranges and kernels grouped by range name, kernel name, and PID/TID. A summary of the kernel instances and their execution times is then computed. The “NVTX Inst” column indicates how many NVTX range instances contained this kernel, while the “Kern Inst” column indicates the number of kernel instances in the summary line.

#### nvtx_pushpop_sum – NVTX Push/Pop Range Summary

Arguments - None

Output: All time values given in nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of NV Tools Extensions Push/Pop Ranges and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### nvtx_pushpop_trace – NVTX Push/Pop Range Trace

Arguments - None

Output: All time values default to nanoseconds

- Start : Range start timestamp

- End : Range end timestamp

- Duration : Range duration

- DurChild : Duration of all child ranges

- DurNonChild : Duration of this range minus child ranges

- Name : Name of the NVTX range

- PID : Process ID

- TID : Thread ID

- Lvl : Stack level, starts at 0

- NumChild : Number of children ranges

- RangeId : Arbitrary ID for range

- ParentId : Range ID of the enclosing range

- RangeStack : Range IDs that make up the push/pop stack

- NameTree : Range name prefixed with level indicator

This report provides a trace of NV Tools Extensions Push/Pop Ranges, their execution time, stack state, and relationship to other push/pop ranges.

#### nvtx_startend_sum – NVTX Start/End Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of NV Tools Extensions Start/End Ranges and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### nvtx_sum – NVTX Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Style : Range style; Start/End or Push/Pop

- Range : Name of the range

This report provides a summary of NV Tools Extensions Start/End and Push/Pop Ranges, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### nvvideo_api_sum – NvVideo API Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this function

- Num Calls : Number of calls to this function

- Avg : Average execution time of this function

- Med : Median execution time of this function

- Min : Smallest execution time of this function

- Max : Largest execution time of this function

- StdDev : Standard deviation of the time of this function

- Event Type : Which API this function belongs to

- Name : Name of the function

This report provides a summary of NvVideo API functions and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that function’s percent of the execution time of the functions listed, and not a percentage of the application wall or CPU execution time.

#### openacc_sum – OpenACC Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of event type

- Count : Number of event type

- Avg : Average execution time of event type

- Med : Median execution time of event type

- Min : Smallest execution time of event type

- Max : Largest execution time of event type

- StdDev : Standard deviation of execution time of event type

- Name : Name of the event

This report provides a summary of OpenACC events and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that event type’s percent of the execution time of the events listed, and not a percentage of the application wall or CPU execution time.

#### opengl_khr_gpu_range_sum – OpenGL KHR_debug GPU Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of OpenGL KHR_debug GPU PUSH/POP debug Ranges, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### opengl_khr_range_sum – OpenGL KHR_debug Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of OpenGL KHR_debug CPU PUSH/POP debug Ranges, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### openmp_sum – OpenMP Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of event type

- Count : Number of event type

- Avg : Average execution time of event type

- Med : Median execution time of event type

- Min : Smallest execution time of event type

- Max : Largest execution time of event type

- StdDev : Standard deviation of execution time of event type

- Name : Name of the event

This report provides a summary of OpenMP events and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that event type’s percent of the execution time of the events listed, and not a percentage of the application wall or CPU execution time.

#### osrt_sum – OS Runtime Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this function

- Num Calls : Number of calls to this function

- Avg : Average execution time of this function

- Med : Median execution time of this function

- Min : Smallest execution time of this function

- Max : Largest execution time of this function

- StdDev : Standard deviation of execution time of this function

- Name : Name of the function

This report provides a summary of operating system functions and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that function’s percent of the execution time of the functions listed, and not a percentage of the application wall or CPU execution time.

#### syscall_sum – Syscall Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this syscall

- Num Calls : Number of calls to this syscall

- Avg : Average execution time of this syscall

- Med : Median execution time of this syscall

- Min : Smallest execution time of this syscall

- Max : Largest execution time of this syscall

- StdDev : Standard deviation of execution time of this syscall

- Name : Name of the syscall

This report provides a summary of syscalls and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that syscall’s percent of the execution time of the syscalls listed, and not a percentage of the application wall or CPU execution time.

#### um_cpu_page_faults_sum – Unified Memory CPU Page Faults Summary

Arguments - None

Output:

> CPU Page Faults : Number of CPU page faults that occurred CPU Instruction Address : Address of the CPU instruction that caused the CPU page faults
>
> This report provides a summary of CPU page faults for unified memory.

#### um_sum\[:rows=\<limit\>\] – Unified Memory Analysis Summary

Arguments

- rows=\<limit\> - Maximum number of rows returned by the query. Default is 10.

Output:

- Virtual Address : Virtual base address of the page(s) being transferred

- HtoD Migration Size : Bytes transferred from Host to Device

- DtoH Migration Size : Bytes transferred from Device to Host

- CPU Page Faults : Number of CPU page faults that occurred for the virtual base address

- GPU Page Faults : Number of GPU page faults that occurred for the virtual base address

- Migration Throughput : Bytes transferred per second

This report provides a summary of data migrations for unified memory.

#### um_total_sum – Unified Memory Totals Summary

Arguments - None

Output:

- Total HtoD Migration Size : Total bytes transferred from host to device

- Total DtoH Migration Size : Total bytes transferred from device to host

- Total CPU Page Faults : Total number of CPU page faults that occurred

- Total GPU Page Faults : Total number of GPU page faults that occurred

- Minimum Virtual Address : Minimum value of the virtual address range for the pages transferred

- Maximum Virtual Address : Maximum value of the virtual address range for the pages transferred

This report provides a summary of all the page faults for unified memory.

#### vulkan_api_sum – Vulkan API Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all executions of this function

- Num Calls: Number of calls to this function

- Avg : Average execution time of this function

- Med : Median execution time of this function

- Min : Smallest execution time of this function

- Max : Largest execution time of this function

- StdDev : Standard deviation of the time of this function

- Name : Name of the function

This report provides a summary of Vulkan API functions and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that function’s percent of the execution time of the functions listed, and not a percentage of the application wall or CPU execution time.

#### vulkan_api_trace – Vulkan API Trace

Arguments - None

Output: All time values default to nanoseconds

- Start : Timestamp when API call was made

- Duration : Length of API calls

- Name : API function name

- Event Class : Vulkan trace event type

- Context : Trace context ID

- CorrID : Correlation used to map to other Vulkan calls

- Pid : Process ID that made the call

- Tid : Thread ID that made the call

- T-Pri : Run priority of call thread

- Thread Name : Name of thread that called API function

This report provides a trace record of Vulkan API function calls and their execution times.

#### vulkan_gpu_marker_sum – Vulkan GPU Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of Vulkan GPU debug markers, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### vulkan_marker_sum – Vulkan Range Summary

Arguments - None

Output: All time values default to nanoseconds

- Time : Percentage of “Total Time”

- Total Time : Total time used by all instances of this range

- Instances : Number of instances of this range

- Avg : Average execution time of this range

- Med : Median execution time of this range

- Min : Smallest execution time of this range

- Max : Largest execution time of this range

- StdDev : Standard deviation of execution time of this range

- Range : Name of the range

This report provides a summary of Vulkan debug markers on the CPU, and their execution times. Note that the “Time” column is calculated using a summation of the “Total Time” column, and represents that range’s percent of the execution time of the ranges listed, and not a percentage of the application wall or CPU execution time.

#### wddm_queue_sum – WDDM Queue Utilization Summary

Arguments - None

Output: All time values default to nanoseconds

- Utilization : Percent of time when queue was not empty

- Instances : Number of events

- Avg : Average event duration

- Med : Median event duration

- Min : Minimum event duration

- Max : Maximum event duration

- StdDev : Standard deviation of event durations

- Name : Event name

- Q Type : Queue type ID

- Q Name : Queue type name

- PID : Process ID associated with event

- GPU ID : GPU index

- Context : WDDM context of queue

- Engine : Engine type ID

- Node Ord : WDDM node ordinal ID

This report provides a summary of the WDDM queue utilization. The utilization is calculated by comparing the amount of time when the queue had one or more active events to total duration, as defined by the minimum and maximum event time for a given Process ID (regardless of the queue context).

### Report Formatters Shipped With Nsight Systems

The following formats are available in Nsight Systems

#### Column

Usage:

``` text
column[:nohdr][:nolimit][:nofmt][:<width>[:<width>]...]
```

Arguments

- `nohdr` : Do not display the header.

- `nolimit` : Remove 100 character limit from auto-width columns Note: This can result in extremely wide columns.

- `nofmt` : Do not reformat numbers.

- `<width>...` : Define the explicit width of one or more columns. If the value `.` is given, the column will auto-adjust. If a width of 0 is given, the column will not be displayed.

The column formatter presents data in vertical text columns. It is primarily designed to be a human-readable format for displaying data on a console display.

Text data will be left-justified, while numeric data will be right-justified. If the data overflows the available column width, it will be marked with a “…” character, to indicate the data values were clipped. Clipping always occurs on the right-hand side, even for numeric data.

Numbers will be reformatted to make easier to visually scan and understand. This includes adding thousands-separators. This process requires that the string representation of the number is converted into its native representation (integer or floating point) and then converted back into a string representation to print. This conversion process attempts to preserve elements of number presentation, such as the number of decimal places, or the use of scientific notation, but the conversion is not always perfect (the number should always be the same, but the presentation may not be). To disable the reformatting process, use the argument `nofmt`.

If no explicit width is given, the columns auto-adjust their width based off the header size and the first 100 lines of data. This auto-adjustment is limited to a maximum width of 100 characters. To allow larger auto-width columns, pass the initial argument nolimit. If the first 100 lines do not calculate the correct column width, it is suggested that explicit column widths be provided.

#### Table

Usage:

``` text
table[:nohdr][:nolimit][:nofmt][:<width>[:<width>]...]
```

Arguments

- `nohdr` : Do not display the header.

- `nolimit` : Remove 100 character limit from auto-width columns Note: This can result in extremely wide columns.

- `nofmt` : Do not reformat numbers.

- `<width>...` : Define the explicit width of one or more columns. If the value `.` is given, the column will auto-adjust. If a width of 0 is given, the column will not be displayed.

The table formatter presents data in vertical text columns inside text boxes. Other than the lines between columns, it is identical to the column formatter.

#### CSV

Usage:

``` text
csv[:nohdr]
```

Arguments

- `nohdr` : Do not display the header.

The csv formatter outputs data as comma-separated values. This format is commonly used for import into other data applications, such as spread-sheets and databases.

There are many different standards for CSV files. Most differences are in how escapes are handled, meaning data values that contain a comma or space.

This CSV formatter will escape commas by surrounding the whole value in double-quotes.

#### TSV

Usage:

``` text
tsv[:nohdr][:esc]
```

Arguments

- `nohdr` : Do not display the header.

- `esc` : escape tab characters, rather than removing them.

The TSV formatter outputs data as tab-separated values. This format is sometimes used for import into other data applications, such as spreadsheets and databases.

Most TSV import/export systems disallow the tab character in data values. The formatter will normally replace any tab characters with a single space. If the esc argument has been provided, any tab characters will be replaced with the literal characters “t”.

#### JSON

Usage:

``` text
json
```

Arguments: no arguments

The JSON formatter outputs data as an array of JSON objects. Each object represents one line of data, and uses the column names as field labels. All objects have the same fields. The formatter attempts to recognize numeric values, as well as JSON keywords, and converts them. Empty values are passed as an empty string (and not nil, or as a missing field).

At this time the formatter does not escape quotes, so if a data value includes double-quotation marks, it will corrupt the JSON file.

#### HDoc

``` text
hdoc[:title=<title>][:css=<URL>]
```

Arguments:

- `title` : string for HTML document title.

- `css` : URL of CSS document to include.

The HDoc formatter generates a complete, verifiable (mostly), standalone HTML document. It is designed to be opened in a web browser, or included in a larger document via an `<iframe>`.

#### HTable

Usage:

``` text
htable
```

Arguments: no arguments

The HTable formatter outputs a raw HTML `<table>` without any of the surrounding HTML document. It is designed to be included into a larger HTML document. Although most web browsers will open and display the document, it is better to use the HDoc format for this type of use.

## Expert Systems Analysis

The Nsight Systems expert system is a feature aimed at automatic detection of performance optimization opportunities in an application’s profile. It uses a set of predefined rules to determine if the application has known bad patterns.

### Using Expert System from the CLI

usage:

``` cpp
nsys [global-options] analyze [options]
       [nsys-rep-or-sqlite-file]
```

If a .nsys-rep file is given as the input file and there is no .sqlite file with the same name in the same directory, it will be generated.

Note

The Expert System view in the GUI will give you the equivalent command line.

### Using Expert System from the GUI

The Expert System View can be found in the same drop-down as the Events View. If there is no .sqlite file with the same name as the .nsys-rep file in the same directory, it will be generated.

The Expert System View has the following components:

1.  Drop-down to select the rule to be run.

2.  Rule description and advice summary.

3.  CLI command that will give the same result.

4.  Table containing results of running the rule.

5.  Settings button that allows users to specify the rule’s arguments.



A context menu is available to correlate the table entry with the timeline. The options are the same as the Events View:

- Zoom to Selected on Timeline (ctrl+double-click)

The highlighting is not supported for rules that do not return an event but rather an arbitrary time range (e.g., GPU utilization rules).

The CLI and GUI share the same rule scripts and messages. There might be some formatting differences between the output table in GUI and CLI.

### Expert System Rules

Rules are scripts that run on the SQLite DB output from Nsight Systems to find common improvable usage patterns.

Each rule has an advice summary with explanation of the problem found and suggestions to address it. Only the top 50 results are displayed by default.

There are currently six rules in the expert system. They are described below. Additional rules will be made available in a future version of Nsight Systems.

#### CUDA Synchronous Operation Rules

**Asynchronous memcpy with pageable memory**

This rule identifies asynchronous memory transfers that end up becoming synchronous if the memory is pageable. This rule is not applicable for Nsight Systems Embedded Platforms Edition

Suggestion: If applicable, use pinned memory instead

> 

**Synchronous Memcpy**

This rule identifies synchronous memory transfers that block the host.

Suggestion: Use cudaMemcpy\*Async APIs instead.

**Synchronous Memset**

This rule identifies synchronous memset operations that block the host.

Suggestion: Use cudaMemset\*Async APIs instead.

**Synchronization APIs**

This rule identifies synchronization APIs that block the host until all issued CUDA calls are complete.

Suggestions: Avoid excessive use of synchronization. Use asynchronous CUDA event calls, such as cudaStreamWaitEvent and cudaEventSynchronize, to prevent host synchronization.

#### GPU Low Utilization Rules

Nsight Systems determines GPU utilization based on API trace data in the collection. Current rules consider CUDA, Vulkan, DX12, and OpenGL API use of the GPU.

**GPU Starvation**

This rule identifies time ranges where a GPU is idle for longer than 500ms. The threshold is adjustable.

Suggestions: Use CPU sampling data, OS Runtime blocked state backtraces, and/or OS Runtime APIs related to thread synchronization to understand if a sluggish or blocked CPU is causing the gaps. Add NVTX annotations to CPU code to understand the reason behind the gaps.

Notes: For each process, each GPU is examined, and gaps are found within the time range that starts with the beginning of the first GPU operation on that device and ends with the end of the last GPU operation on that device. GPU gaps that cannot be addressed by the user are excluded. This includes:

- Profiling overhead in the middle of a GPU gap.

- The initial gap in the report that is seen before the first GPU operation.

- The final gap that is seen after the last GPU operation.

**GPU Low Utilization**

This rule identifies time regions with low utilization.

Suggestions: Use CPU sampling data, OS Runtime blocked state backtraces, and/or OS Runtime APIs related to thread synchronization to understand if a sluggish or blocked CPU is causing the gaps. Add NVTX annotations to CPU code to understand the reason behind the gaps.

Notes: For each process, each GPU is examined, and gaps are found within the time range that starts with the beginning of the first GPU operation on that device and ends with the end of the last GPU operation on that device. This time range is then divided into equal chunks, and the GPU utilization is calculated for each chunk. The utilization includes all GPU operations as well as profiling overheads that the user cannot address.

The utilization refers to the “time” utilization and not the “resource” utilization. This rule attempts to find time gaps when the GPU is or isn’t being used, but does not take into account how many GPU resources are being used. Therefore, a single running memcpy is considered the same amount of “utilization” as a huge kernel that takes over all the cores. If multiple operations run concurrently in the same chunk, their utilization will be added up and may exceed 100%.

Chunks with an in-use percentage less than the threshold value are displayed. If consecutive chunks have a low in-use percentage, the individual chunks are coalesced into a single display record, keeping the weighted average of percentages. This is why returned chunks may have different durations.

## Advanced Report Analysis

Nsight Systems Advanced Report Analysis is functionality to better support complex statistical analysis across multiple result files. Possible use cases for this functionality include:

- Multi-Node Analysis - When you run Nsight Systems across a cluster, it typically generates one result file per rank on the cluster. While you can load multiple result files into the GUI for visualization, this analysis system allows you to run statistical analysis across all of the result files.

- Multi-Pass Analysis - Some features in Nsight Systems cannot be run together due to overhead or hardware considerations. For example, there are frequently more CPU performance counters available than your CPU has registers. Using this analysis, you could run multiple runs with different sets of counters and then analyze the results together.

- Multi-Run Analysis - Sometimes you want to compare two runs that were not taken at the same time together. Perhaps you ran the tool on two different hardware configurations and want to see what changed. Perhaps you are doing regression testing or performance improvement analysis and want to check your status. Comparing those result files statistically can show patterns.

- Complex/multi-phase analysis - Sometimes you may want to perform a complicated, or multi-phase analysis on one or more results files. The helper functionality available in the Advanced Analysis system can simplify common steps.

- Complex data output - Sometimes you want to be able to build complex visualizations from your analysis, rather than just tabular data from bare statistics.

**Analysis Steps**

Note

Prior to using advanced analysis, please make sure that you have installed all required dependencies. See **Installing Advanced Analysis System** in the **Installation Guide** for more information.

1.  Generate the reports - Generate the reports as you always have, in fact, you can use reports that you have generated previously.

2.  Set up - Choose the recipe (See Available Recipes, below), give it any required parameters, and run.

3.  Launch Analysis - Nsight Systems will run the analysis, using your local system or Dask, as you have selected.

4.  Output - the output is a directory containing an .nsys-analysis file, which can then be opened within the Nsight Systems GUI.

5.  View the data - depending on your recipe, you can have any number of visualizations, from simple tabular information to Jupyter notebooks which can be opened inside the GUI.

### Available Advanced Analysis Recipes

All advanced analysis recipes are run using the `recipe` CLI command switch.

usage:

``` text
nsys recipe [args] <recipe-name> [recipe args]
```

Nsight Systems provides several initial analysis recipes, mostly based around making our existing statistics and expert systems rules run multi-report.

These recipes can be found at `<target-linux-x64>/python/packages/nsys-recipe/recipes`. Please note that all recipes are in the form of python scripts. You may alter the given recipes or write your own to meet your needs. Refer to [Tutorial: Create a User-Defined Recipe](#create-user-defined-recipe) for an example of how to do this. However, be advised that the APIs may change for the next few versions. Additional recipes will be added on an ongoing basis.

For more information about a specific recipe, including recipe parameters, please use `nsys`` ``recipe`` ``[recipe`` ``name]`` ``--help`.

**List of recipes**

Each recipe will be tagged with one or more keywords to help understand its purpose.

| Keywords | Description |  |
|----|----|----|
| Expert System | The recipe originated from the Expert System. A script with the same name is also available via `nsys`` ``analyze`, but its behavior and implementation may differ. |  |
| Stats System | The recipe originated from the Stats System. A script with the same name is also available via `nsys`` ``stats`, but its behavior and implementation may differ. |  |
| Trace | The recipe provides a trace record of individual events that are observable in the GUI timeline. |  |
| Summary | The recipe provides a summarized view of events, often representing aggregated data. |  |
| Pace | The recipe provides a detailed analysis of how a specific event progresses across the application. |  |
| Heatmap | The recipe provides a heatmap that visualizes patterns across the application. |  |

- cuda_api_sumCUDA API Summary  
  This recipe provides a summary of CUDA API functions and their execution times.

  Keywords: CUDA, Summary, Stats System

- cuda_api_syncCUDA Synchronization APIs  
  This recipe identifies synchronization APIs that block the host until the issued CUDA calls are complete.

  Keywords: CUDA, Synchronization, Trace, Expert System

- cuda_gpu_kern_histCUDA GPU Kernel Duration Histogram  
  This recipe represents the probability of the duration of a CUDA kernel among all its instances or all kernels in the program.

  Keywords: CUDA, Kernel, Histogram, Duration

- cuda_gpu_kern_paceCUDA GPU Kernel Pacing  
  This recipe investigates the progress and consistency of a particular CUDA kernel throughout the application.

  Keywords: CUDA, Kernel, Pace

- cuda_gpu_kern_sumCUDA GPU Kernel Summary  
  This recipe provides a summary of CUDA kernels and their execution times.

  Keywords: CUDA, Kernel, Summary, Stats System

- cuda_gpu_mem_size_sumCUDA GPU MemOps Summary (by Size)  
  This recipe provides a summary of GPU memory operations and the amount of memory they utilize.

  Keywords: CUDA, Memory, Summary, Stats System

- cuda_gpu_mem_time_sumCUDA GPU MemOps Summary (by Time)  
  This recipe provides a summary of GPU memory operations and their execution times.

  Keywords: CUDA, Memory, Summary, Stats System

- cuda_gpu_time_util_mapCUDA GPU Time Utilization Heatmap  
  This recipe calculates the percentage of time that CUDA kernels were running.

  Keywords: CUDA, Kernel, Heatmap

- cuda_memcpy_asyncCUDA Async Memcpy with Pageable Memory  
  This recipe identifies asynchronous memory transfers that end up becoming synchronous if the memory is pageable.

  Keywords: CUDA, Memcpy, Trace, Expert System

- cuda_memcpy_syncCUDA Synchronous Memcpy  
  This recipe identifies memory transfers that are synchronous.

  Keywords: CUDA, Memcpy, Trace, Expert System

- cuda_memset_syncCUDA Synchronous Memset  
  This recipe identifies synchronous memset operations with pinned host memory or Unified Memory region.

  Keywords: CUDA, Memset, Trace, Expert System

- diffStatistics Diff  
  This script compares outputs from two runs of the same statistical recipe.

  Keywords: Diff, Summary

- dx12_mem_opsDX12 Memory Operations  
  This recipe flags problematic memory operations with warnings.

  Keywords: DX12, Memory, Trace, Expert System

- file_access_sumOS Runtime File Access Summary  
  This recipe provides a summary of file access functions, including high-level overview of file access patterns across the system.

  For details and use cases of this recipe, see [file_access_sum Recipe](#file-access-sum-recipe).

  Keywords: OSRT, Summary

- gfx_hotspotGraphics Hotspot Analysis  
  This recipe generates a report of CPU hotspots for graphics applications.

  The output format for this recipe is different than other recipes. See [gfx_hotspot Recipe](#gfx-hotspot-recipe).

  Keywords: DX12, Vulkan, Summary, Trace

- gpu_gapsGPU Gaps  
  This recipe identifies time regions where a GPU is idle for longer than a set threshold.

  Keywords: CUDA, Utilization, Expert System

- gpu_metric_util_mapGPU Metric Utilization Heatmap  
  This recipe calculates the percentage of SM Active, SM Issue, and Tensor Active metrics.

  Keywords: GPU Metrics, Heatmap

- gpu_metric_util_sumGPU Metrics Utilization Summary  
  This recipe provides a summary of different GPU metrics. GPU metrics are based on binary inclusion. Any ranges that do not include at least one sampling point are excluded from the output.

  Keywords: GPU Metrics, Summary

- gpu_time_utilGPU Time Utilization  
  This recipe identifies time regions with low GPU utilization.

  Keywords: CUDA, Utilization, Expert System

- gpu_vram_usage_traceGPU VRAM Usage Trace  
  This recipe traces the VRAM usage of GPU workloads, allowing comparison of changes between CPU frames, and identifying issues in resource migration between VRAM and SYSMEM, and with resource allocation & deallocation.

  For details and use cases of this recipe, see [gpu_vram_usage_trace Recipe - Preview Feature](#gpu-vram-usage-trace-recipe).

  Keywords: VRAM, Trace

- mpi_gpu_time_util_mapMPI and GPU Time Utilization Heatmap  
  This recipe calculates the percentage of time that CUDA kernels were running and MPI communication was active, as well as their overlap.

  Keywords: MPI, CUDA, Kernel, Utilization, Heatmap

- mpi_sumMPI Summary  
  This recipe provides a summary of MPI functions and their execution times.

  Keywords: MPI, Summary

- nccl_gpu_overlap_traceNCCL GPU Overlap Trace  
  This recipe calculates the percentage of overlap for communication and compute kernels. Communication kernels are identified by the ‘nccl’ prefix.

  Keywords: NCCL, CUDA, Kernel, Overlap, Trace

- nccl_gpu_proj_sumNCCL GPU Projection Summary  
  This recipe provides a summary of NCCL functions projected from the CPU onto the GPU, and their execution times.

  Keywords: NCCL, CUDA, GPU Projection, Summary

- nccl_gpu_time_util_mapNCCL GPU Time Utilization Heatmap  
  This recipe calculates the percentage of time that communication and compute kernels were running, as well as their overlap. Communication kernels are identified by the ‘nccl’ prefix.

  Keywords: NCCL, CUDA, Kernel, Utilization, Overlap, Heatmap

- nccl_sumNCCL Summary  
  This recipe provides a summary of NCCL functions and their execution times.

  Keywords: NCCL, Summary

- network_map_awsAWS Metrics Heatmap  
  This recipe displays heatmaps of AWS EFA metrics.

  Keywords: Network, AWS, EFA, Heatmap

- network_sumNetwork Traffic Summary  
  This recipe provides a summary of the network traffic over NICs and InfiniBand Switches.

  Keywords: Network, Summary

- network_traffic_mapNetwork Devices Traffic Heatmap  
  This recipe displays heatmaps of sent traffic, received traffic, and congestion events for network devices.

  Keywords: Network, Heatmap

- nvtx_cpu_topdownCPU Topdown methodology metrics correlated to NVTX ranges  
  This recipe calculates CPU Topdown methodology metrics for NVTX push/pop ranges based on collected PMU core events for NVIDIA CPUs featuring Arm cores.

  For details and use cases of this recipe, see [nvtx_cpu_topdown Recipe](#nvtx-cpu-topdown-recipe).

  Keywords: NVTX, CPU Topdown, Metrics, Summary

- nvlink_sumNVLink Network Throughput Summary  
  This recipe provides a summary of the NVLink network throughput.

  Keywords: NVLink, Summary

- nvtx_gpu_proj_paceNVTX GPU Projection Pacing  
  This recipe investigates the progress and consistency of a particular NVTX range projected from the CPU onto the GPU, throughout the application.

  Keywords: NVTX, GPU Projection, Pace

- nvtx_gpu_proj_sumNVTX GPU Projection Summary  
  This recipe provides a summary of NVTX time ranges projected from the CPU onto the GPU, and their execution times.

  Keywords: NVTX, GPU Projection, Summary, Stats System

- nvtx_gpu_proj_traceNVTX GPU Projection Trace  
  This recipe provides a trace of NVTX time ranges projected from the CPU onto the GPU.

  Keywords: NVTX, GPU Projection, Trace, Stats System

- nvtx_paceNVTX Pacing  
  This recipe investigates the progress and consistency of a particular NVTX range throughout the application.

  Keywords: NVTX, Pace

- nvtx_sumNVTX Range Summary  
  This recipe provides a summary of NVTX Start/End and Push/Pop Ranges, and their execution times.

  Keywords: NVTX, Summary, Stats System

- osrt_sumOS Runtime Summary  
  This recipe provides a summary of C library functions and their execution times.

  Keywords: OSRT, Summary, Stats System

- s3_access_sumS3 Access Summary  
  This recipe provides a summary of S3 object store access across traced S3 client libraries.

  For details and use cases of this recipe, see [s3_access_sum Recipe](#s3-access-sum-recipe).

  Keywords: S3, Storage, Summary

- storage_util_mapStorage Metrics Heatmap  
  This recipe displays heatmaps of storage devices metrics.

  Keywords: Storage, Heatmap

- ucx_gpu_time_util_mapUCX and GPU Time Utilization Heatmap  
  This recipe calculates the percentage of time that CUDA kernels were running and UCX communication was active, as well as their overlap.

  Keywords: UCX, CUDA, Kernel, Heatmap

### Recipe Output Examples

A successful recipe run outputs a directory containing different files. This section gives some common examples of these output types.

**Table**

Trace or summary data will be stored in data storage formats such as CSV, Parquet, or Arrow. Typically, you can also access the same data within the output Jupyter notebook.

- Summary table:



- Trace table:



- Overlap table:



**Visualization**

Some recipes include data visualization in the output Jupyter notebooks. These graphs use Plotly, which provides interactivity.

- Summary graph:



- Box plot:



- Line graph:



- Top N graph:



- Pace graph:



- Heatmap:



### Opening in Jupyter Notebook

Running the recipe command creates a new analysis file (.nsys-analysis). Open the Nsight Systems GUI and select `File->Open`, and pick your file.



Open the folder icon and click on the notebook icon to open the Jupyter notebook.



Run the Jupyter notebook:



And the output appears on-screen. In this case a heat map of activity running a Jacobi solver.



### Configuring Dask

The multi-report analysis system does not offer options to configure the Dask environment. However, you could achieve this by modifying the recipe script directly or using one of the following from Dask’s configuration system:

- YAML files: By default, Dask searches for all YAML files in `~/.config/dask/` or `/etc/dask/`. This search path can be changed using the environment variable `DASK_ROOT_CONFIG` or `DASK_CONFIG`. See the [Dask documentation](https://docs.dask.org/en/stable/) for the complete list of locations and the lookup order. Example:

  ``` text
  $ cat example.yaml
  'Distributed':
          'scheduler':
              'allowed-failures': 5
  ```

- Environment variables: Dask searches for all environment variables that start with `DASK_`, then transforms keys by converting to lower-case and changing double-underscores to nested structures. See Dask documentation for the complete list of variables. Example:

  ``` text
  DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=5
  ```

**Dask Client**

With no configuration set, the dask-futures mode option initializes the Dask Client with the default arguments, which results in creating a LocalCluster in the background. The following are the YAML/environment variables that could be set to change the default behavior:

- distributed.comm.timeouts.connect / DASK_DISTRIBUTED\_\_COMM\_\_TIMEOUTS\_\_CONNECT

- client-name / DASK_CLIENT_NAME

- scheduler-address / DASK_SCHEDULER_ADDRESS

- distributed.client.heartbeat / DASK_DISTRIBUTED\_\_CLIENT\_\_HEARTBEAT

- distributed.client.scheduler-info-interval / DASK_DISTRIBUTED\_\_CLIENT\_\_SCHEDULER_INFO_INTERVAL

- distributed.client.preload / DASK_DISTRIBUTED\_\_CLIENT\_\_PRELOAD

- distributed.client.preload-argv / DASK_DISTRIBUTED\_\_CLIENT\_\_PRELOAD_ARGV

**Recipe’s environment variables**

Recipe has its own list of environment variables to make the configuration more complete and flexible. These environment variables are either missing from Dask’s configuration system or specific to the recipe system:

- NSYS_DASK_SCHEDULER_FILE: Path to a file with scheduler information. It will be used to initialize the Dask Client.

- NSYS_DIR: Path to the directory of Nsight Systems containing the target and host directories. The nsys executable and the recipe dependencies will be searched in this directory instead of the one deduced from the currently running recipe file path.

### gfx_hotspot Recipe

This recipe’s output is different from other recipes and is presented as a web application.

The output can be viewed by passing the `--run-viewer` argument to the recipe - along with the further `--show-viewer` which will automatically open a web browser to the report view.

Alternatively, a previously-executed `gfx_hotspot` recipe’s output can be viewed by executing the `run_viewer.py` script from the recipe output folder.

For the best results, run the recipe on a report with resolved symbols.

**Threading Analysis**

In this tab, an overview of the multi-threading behavior of the target (most active) process is presented.

- Application Statistics:  
  This table shows the CPU and thread statistics for the target process.

- CPU Info:  
  This table shows information about the CPU hardware.

- Top 5 Processes CPU Utilisation:  
  This table shows the most active processes during the sample, to help detect situations where another process is interfering with the target process’s execution.

- Threading Health Check:  
  This table contains a list of very common CPU-bound application performance indicators. If the target application is GPU bound, the entire table will be shown in green. If it is CPU bound, then each row will be highlighted in green if the value is healthy, in yellow if it requires attention, and in red if it potentially indicates of a threading issue. For unhealthy metrics, the “warning” column will also show steps or investigation angles that may be considered in order to improve the result.

- Thread Utilisation:  
  This graph shows the process threads, ordered from most busy to least.

- Thread Concurrency:  
  This graph shows the percentage and amount of time an average graphic frame is running each number of threads concurrently. High percentage of low thread counts could indicate excessive serialization in the algorithm, where CPU work could be better parallelized by improving the use of multi-threading.

**Hotspot Analysis**

In this tab, frames are selected in one of four methods:  
- Longest Frame time (Slow Frames)

- Periodic time-based selection (Periodic Frames)

- Frames with highest transfer activity (Bar1 Reads)

- Frames with least GPU activity (GR Idle)

The report view then allows comparing the selected frames to each other and to the median frame in the same metric, helping identify the main differences and possible problem areas in each one.

- Overview:  
  These tables show the report overview as well as the frame selection method and other capture-wide statistics and general information. A shorthand list of the “Performance Issues” table for each frame is also shown.

- Frame Times:  
  This graph shows a sequence of the graphical frames (CPU time and GPU time derived from GPU Utilisation percentage per CPU frame time) ordered by their index. The selected frames are indicated and labelled. Clicking any of the indicated frames will set it as the left frame for comparison.

- Region / Compare to:  
  These controls allow selecting the two frames to be shown for comparison. “Periodic Frames” shows 10 sampled frames (with equally distributed indices), while the other three modes show the 5 frames with the highest value in the chosen metric and the median frame in the same metric. All information from this point onwards is shown per selected frame in each of the two columns, allowing for 1-to-1 comparison. Selecting the same frame for both controls will show just the single frame as the entire width of the view.

- Frame Info:  
  This table shows the frame duration and start time, the number of threads that were active during the frame, and the thread IDs of key threads in the frame processing operation which are important for determining likely performance issues.

- Performance Issues:  
  This table shows the key performance limiters and hotspots for the selected frame. Each indicator will have a breakdown of what indicators were present to call out the performance issue during this frame. These indicators are not necessarily the root cause of the problems in the region, but have been flagged for consideration.

- GPU Metrics:  
  This table shows the average or total (respectively) values of the GPU metrics collected during the frame time. If GPU Metrics were not collected, this table will not appear.

- System ETW Events (Windows only):  
  This chart shows a breakdown of the system process-reported ETW events during the frame. If WDDM trace and Custom ETW trace were not collected, this chart will not appear.

- DxgKrnl Events (Windows only):  
  This chart shows a breakdown of the DxgKrnl ETW provider events during the frame. If WDDM trace was not collected, this chart will not appear.

- CPU Thread Utilisation Time:  
  This graph shows the time spent inside each thread during the frame. The bars match the two selected frames, and the matching-colored line shows the total frame time. Clicking any of the columns in the graph will select that thread for the following elements in the report.

- Thread:  
  This control allows selecting the thread to be shown in the following views.

- Call stacks:  
  This control shows the sampled call stacks during the frame. Clicking a call stack frame will filter the view to only show call stacks containing this call stack frame, allowing to drill down into potential problem areas. The title of the control indicates the two modes selected for display, which can be switched with the two toggles in the top right of the control:

- Call stacks - Merged:  
  Merges all similar call stacks logically, regardless of when in the frame time the functions appeared. This is useful to see where the cumulative time is spent.

- Call stacks - Over Time:  
  Keeps call stacks ordered chronologically, so that repeated calls to the same function appear separately.

- Periodic Sampled Call stacks:  
  Only shows call stacks acquired by periodic sampling (matching the orange marks in Nsight Systems’s timeline view). This view provides a better statistical overview of where the frame time was spent.

- All Call stacks:  
  Shows periodic sampled call stacks as well as call stacks acquired from other sources such as call stacks from ETW events (Windows) and event-based sampling (Linux) (matching both the orange AND the grey marks in Nsight Systems’s timeline view).

- Modules in Sampled Call Stacks:  
  This graph shows the number of call stacks in the frame that include at least one call stack frame in a function belonging to each module. This helps identify which modules were the most active during the frame.

- ETW Events (Windows only):  
  This chart shows a breakdown of the thread-reported ETW events during the frame. If WDDM trace and Custom ETW trace were not collected, this chart will not appear.

- Context Switch Call Stacks:  
  This table shows a breakdown of the call stacks that led to context switches for the thread during the frame, indicating where the thread may have stalled. Hovering the mouse cursor over the “Name” column will show the full call stack for each entry.

- DX12 API / Vulkan API:  
  These tables show a breakdown of the graphical API functions that appeared in sampled call stacks. If DX12 / Vulkan trace were not collected, these tables will not appear.

- Known Symbols From Sampled Call Stacks:  
  This table shows a breakdown of known symbols that often cause performance issues, such as DX12’s CreateCommittedResource. If symbols were not resolved for the nsys-rep file, this table will not appear.

- PIX Markers (Windows only):  
  This table shows a breakdown of PIX marker ranges that contained sampled call stacks. If WDDM trace and DX11 / DX12 trace were not collected or the target application does not use PIX markers, this table will not appear.

### gpu_vram_usage_trace Recipe - Preview Feature

This recipe analyzes VRAM usage patterns and statistics from Nsight Systems reports, helping identify and troubleshoot potential issues in GPU memory management.

**Overview**

The gpu_vram_usage_trace recipe generates an interactive Jupyter notebook that analyzes VRAM usage patterns, such as resource migrations between VRAM and system memory, and resource allocation and deallocation timing. This analysis helps identify memory management issues, optimization opportunities, and potential causes of performance degradation related to GPU memory usage.

**Key Capabilities**

The recipe provides insights into:

- **VRAM Usage Tracking**: Per-frame monitoring of VRAM’s and SYSMEM’s usage, commitment, and budget across all GPU resources.

- **Memory Resource Details**: Comprehensive information for all allocated GPU memory resources.

- **Resource Migrations**: Analysis of resource migration patterns between VRAM and system memory.

- **Frame-by-Frame Analysis & Comparison**: Detailed view of available resources at a specific point in time and comparison to another.

- **Debugging Context Integration**: Correlation of memory resource usage with user-provided performance markers, resource debug names, and callstack information.

**Use Cases**

The recipe is particularly valuable for identifying and addressing the following scenarios (but not limited to these):

1.  **VRAM Exhaustion**: Detecting when applications approach or exceed available VRAM limits, and identifying which resources are consuming the available VRAM.

2.  **Memory Thrashing**: Identifying excessive resource migrations between VRAM and system memory.

3.  **Frame Spikes**: Analyzing frames with abnormal performance due to VRAM usage and/or resource transitions.

**Important Notes**

- **Preview Feature**: This recipe is a preview feature and may be subject to change in the near future.

- **Single Report Only**: This recipe is intended for use with a single report. Using it with multiple reports may cause unexpected behavior.

**Prerequisites**

- **Windows (DirectX 12 and Vulkan)**: This recipe currently supports reports recorded on Windows, using either DirectX 12 or Vulkan.

**Usage**

This recipe requires that Nsight Systems reports be collected with WDDM tracing enabled.

Steps:

1.  **Collect an nsys-rep report with WDDM tracing enabled**

    - **With \`nsys.exe\` (CLI)**: Use the parameters –trace=wddm together with either –wddm-memory-trace=true or –wddm-additional-events=true.

    - **With \`nsys-ui.exe\` (GUI)**: Enable the **WDDM Trace** collector, using either the “Collect WDDM memory trace” or “Extensive trace including Hardware Scheduling queues…” option.

    Note

    *Optional: To collect additional debug name information on resources (affects the Resident Resources Details section), enable tracing of debug markers as follows:*

    - **For DirectX 12:**

      - With **\`nsys-ui.exe\` (GUI)**: Enable the **DX12 Trace** collector and the **Trace Debug Markers** option.

      - With **\`nsys.exe\` (CLI)**: Adjust the trace argument: `--trace=wddm,dx12-annotations`.

    - **For Vulkan:**

      - With **\`nsys-ui.exe\` (GUI)**: Enable the **Vulkan Trace** collector and the **Trace Debug Markers** option.

      - With **\`nsys.exe\` (CLI)**: Adjust the trace argument: `--trace=wddm,vulkan-annotations`.

2.  **Run the recipe** `nsys`` ``recipe`` ``gpu_vram_usage_trace`` ``--input`` ``[report`` ``file`` ``path]`

3.  **Open the generated notebook** Open the produced stats.ipynb Jupyter notebook to view the interactive analysis.

**Output**

As the main output, the recipe generates an interactive Jupyter notebook `stats.ipynb` with the following sections:

- Global Process and GPU Selectors:  
  

- Interactive Timeline Charts:  
  

- Resident Resources Diff Tables:  
  

- Resident Resources Details Section:  
  

- All Allocations Table:  
  

**Recommended Workflow**

- Start by selecting the process and GPU of interest using the global selectors at the top.

- Identify and select suspicious frames, such as frames with unusual memory usage or with a high volume of memory transition events.

- Use the Resident Resources table to learn more about allocated resources during the selected frames:  
  - Identify resources with significant memory usage.

  - Identify resources that have changed between the two points in time (transitioned between VRAM and SYSMEM, or were allocated/deallocated).

  - View resources’ allocation details, performance markers, and callstacks to help recognize the specific resources and possible problematic settings.

- Use the “All Allocations” table to find suspicious resources, such as resources with excessive residency changes or other unusual characteristics.

### nvtx_cpu_topdown Recipe

This recipe calculates CPU Topdown methodology metrics for NVTX push/pop ranges based on collected PMU core events for NVIDIA CPUs featuring Arm cores. It can process multiple Nsight Systems reports.

Currently, the recipe supports NVIDIA Grace (TM) CPUs and NVIDIA DGX Spark (TM) CPUs.

We recommend using this recipe after running the `collect_cpu_topdown.sh` script, which simplifies collecting all PMU core event and metric data needed to perform a CPU Topdown analysis of the workload’s CPU performance. For more details on this script, refer to the [Arm Topdown Analysis](../UserGuide/index.html#arm-topdown-analysis) section.

If PMU core events other than those required by Topdown are collected, the recipe will calculate available CPU metrics based on them and display those metrics in the output.

**Use Case**

The recipe is most useful when the following conditions are met:

1.  The application runs on CPU cores supported by the recipe.

2.  The application is instrumented with NVTX push/pop ranges.

3.  NVTX range spans a specific CPU algorithm / code section that does not make syscalls or calls to other libraries whose functions take significant time to execute.

4.  NVTX ranges with the same name are used to represent the same workload across all threads and all repetitions.

5.  The duration of the NVTX range is 5 ms or longer to obtain more accurate results.

6.  In systems with heterogeneous CPU cores, the NVTX range executed on a given core type remains consistent across runs, ensuring a reliable view of CPU metrics for that range on that core type. For example, to achieve stable runs, you can pin the process to specific CPU cores using `taskset`.

Note

For the case of NVTX ranges from multiple threads, only the NVTX ranges from either the main thread (default) or the thread specified via `--thread-name` will be processed.

**Usage**

``` text
[1] mkdir reports && cd reports
[2] <path to target-linux-sbsa-armv8>/CpuProfiling/collect_cpu_topdown.sh ./myApp
[3] nsys recipe nvtx_cpu_topdown --input .
```

1.  This step creates a new directory to store the reports. We recommend using an empty new directory, because the `collect_cpu_topdown.sh` script overwrites the output files and does not currently allow customization of file names.

2.  This step creates several report files: cpu-td1.nsys-rep, cpu-branch-ipc.nsys-rep, etc.

    Note

    Note that since multiple reports are created, this step can take significant time to complete.

3.  This step runs the recipe, uses all reports in the current directory as the input, and produces a `.ipynb` Jupyter notebook, `.parquet` and `.csv` (if `--csv` is specified) files as the output.

**Output**

As the main output, the recipe generates the Jupyter notebook `nvtx_cpu_topdown.ipynb` with the following sections:

- NVTX Summary for Heterogeneous CPU Cores:  
  Displays a summary of NVTX ranges compiled from Nsight Systems reports provided to the recipe.

  For a report selected from the drop-down menu, the section shows each NVTX range (in call stack order) with its instance count, median\* instance duration, and CPU time aggregated across heterogeneous cores, as well as CPU time per core type - both related to the NVTX instance with the median\* duration.

  Note: median\* is defined as the middle value in the sorted list. For an even number of elements, it is the second of the two middle values.

  Note

  This section is available only for data collected from heterogeneous CPU cores. For these cores, the remaining sections apply to each CPU core type individually and can be toggled using the `Select`` ``CPU`` ``Core` drop-down menu.

  

- Warnings:  
  Displays warnings generated during recipe execution and related to the entire recipe output (or to the portion of it specific to a given CPU core type). If there are no warnings, this section is not displayed.

- NVTX Summary:  
  Displays a summary of NVTX ranges compiled from Nsight Systems reports provided to the recipe.

  For a report selected from the drop-down menu, the section shows each NVTX range (in call stack order) with its instance count, median duration with median absolute deviation, median CPU time with median absolute deviation, and relevant notes.

  If NVTX ranges are filtered out, they are grayed out in the table, and a note is displayed in the Notes column for the corresponding range. The following ranges are candidates to be filtered out:

  - Ranges that contain fewer than 3 PMU samples in at least one Nsight Systems report provided to the recipe.

  - Ranges that are not present in at least one Nsight Systems report provided to the recipe.

  If NVTX ranges are not stable across some of the reports, the section will display a warning next to the unstable data and a note in the Notes column for the corresponding range.

  

- CPU Topdown Methodology Metrics:  
  Presents the metric results of the CPU Topdown methodology for the selected NVTX range.

  For the range name selected from the drop-down menu, the most appropriate NVTX range instance is identified from the Nsight Systems reports as follows:

  > **For data collected from heterogeneous CPU cores:** The NVTX range instance with the median\* CPU time is selected from each report.
  >
  > **Otherwise:** The NVTX range instance with the median\* duration is selected from the first report (displayed by default in the NVTX Summary section). The corresponding instance index is then used to extract data from subsequent reports.
  >
  > Note: median\* is defined as the middle value in a sorted list. For an even number of elements, it is the second of the two middle values.

  The section shows the following tables:

  1.  Topdown Level 1 metrics

  2.  Frontend Bound metrics

  3.  Backend Bound metrics

  4.  Bad Speculation metrics

  5.  Retiring metrics

  6.  Miscellaneous metrics

  Each table is displayed only when the required data is available.

  

- Report Summary:  
  Displays information about the Nsight Systems report files given to the recipe for input, as well as: the PMU core events collected in each specific report, and the CPU core metrics computed for each specific report.

  

### file_access_sum Recipe

This recipe analyzes file access patterns and I/O performance statistics from one or more Nsight Systems reports, aggregating data across processes and machines.

**Overview**

The file_access_sum recipe generates an interactive Jupyter notebook that analyzes POSIX VFS (Virtual File System) function calls captured during profiling sessions. This analysis helps identify I/O bottlenecks, optimization opportunities, and file access patterns that could impact application performance.

**Key Capabilities**

The recipe provides insights into:

- **File Access Patterns**: Breakdown of read-only, write-only, and read-write file access patterns.

- **Performance Metrics**: Total bytes transferred, operation counts, and average I/O sizes per operation.

- **Cross-Process Analysis**: File access patterns across multiple hosts, processes, and threads.

- **Temporal Analysis**: Distribution of CPU time by operation type.

- **Hotspot Identification**: Top files by read/write volume and operation frequency.

- **Performance Recommendations**: Automated detection of potentially inefficient I/O patterns with actionable suggestions.

**Use Cases**

The recipe is particularly valuable for identifying and addressing the following scenarios (but not limited to these):

1.  **I/O Patterns**: Understanding application I/O behavior to uncover usage trends and inefficiencies.

2.  **Small I/O Operations**: Detection of frequent small read/write operations that could benefit from batching.

3.  **Caching Opportunities**: Identification of frequently accessed read-only files that are candidates for local caching.

4.  **Metadata Contention**: Identifying cases where frequent metadata operations by one process may cause contention, impacting storage access for other processes.

5.  **System File Noise**: Filtering out system files (/dev/, /sys/, etc.) to focus on application-relevant I/O.

**Prerequisites**

This recipe requires that Nsight Systems reports be collected with specific tracing parameters:

- `--trace=osrt` - Enables OS Runtime API tracing

- `--osrt-file-access=true` - Enables file access tracking

- **Optional:** To enable tracing of MPI rank information, use `--trace=mpi` along with either `--mpi-impl=openmpi` or `--mpi-impl=mpich`.

- **Optional:** To enable the NVTX range correlation table, instrument your application code with NVTX ranges. See [Marking and Labeling Regions](../UserGuide/index.html#marking-and-labeling-regions) in the User Guide.

**Usage**

``` text
[1] Create a reports folder.
[2] Collect nsys-rep reports, using '--trace=osrt' and '--osrt-file-access=true' parameters, and save them to the reports folder.
[3] Run the recipe, using 'nsys recipe file_access_sum --input [reports folder path]'.
```

**Output**

As the main output, the recipe generates an interactive Jupyter notebook `file_access_stats.ipynb` with the following sections:

- File Access Summary Table:  
  

- Hottest Read/Write Files Tables:  
  

- All Files Table:  
  

- Read/Write Access Histogram:  
  

- CPU Time Graph:  
  

- Operations Count Chart:  
  

- Performance Analysis:  
  

- NVTX Ranges Analysis:  
  

**Recommended Workflow**  
- Start by setting file path ignore patterns to exclude system files from analysis.

- Focus on application-specific files during the analysis by using regex filtering.

- View the Nsight Systems report file alongside this analysis to gain a deeper understanding of the application’s behavior.

### s3_access_sum Recipe

This recipe analyzes S3 access patterns and I/O statistics from one or more Nsight Systems’ reports, aggregating data across processes and hosts.

**Overview**

The s3_access_sum recipe generates an interactive Jupyter notebook that analyzes S3 operations captured during profiling sessions. It provides a statistical overview of which buckets and objects were accessed and of access patterns.

**Key Capabilities**

The recipe provides insights into:

- **Bucket-Level Summary**: Aggregated access statistics per S3 bucket, including object counts, download/upload bytes, and operation counts.

- **Object Access Patterns**: Breakdown of download-only, upload-only, and mixed-access objects.

- **Hotspot Identification**: Top objects by download/upload volume and operation frequency.

- **Cross-Process Analysis**: S3 access patterns across multiple hosts, processes, and threads.

- **NVTX Range Correlation**: S3 activity correlated with user-defined NVTX ranges, such as access-time statistics and I/O volume.

**Use Cases**

The recipe is particularly valuable for identifying and addressing the following scenarios (but not limited to these):

1.  **Hot Buckets and Objects**: Identifying which S3 buckets and objects carry the most traffic to focus optimization efforts.

2.  **Small or Frequent Operations**: Detecting frequent small S3 transfers that could benefit from batching or prefetching.

3.  **Application Phase Correlation**: Using NVTX ranges to understand which application phases drive the most S3 I/O.

4.  **Multi-Rank Comparison**: Comparing S3 usage across MPI ranks or hosts to identify load imbalances.

**Supported S3 Client Libraries**

The recipe counts the following operations from traced S3 client libraries:

- **AWS CRT** (aws-c-s3): `GetObject`, `PutObject`, `CopyObject`

- **AWS C++ SDK** (aws-cpp-sdk-s3): `GetObject`, `PutObject`, `CopyObject`, `DeleteObject`, `HeadObject`, `CreateMultipartUpload`, `UploadPart`, `CompleteMultipartUpload`, `AbortMultipartUpload`, `ListObjects`, `ListObjectsV2`, `CreateBucket`

- **Boto3**: `get_object`, `put_object`, `head_object`, `upload_part`, `create_multipart_upload`, `abort_multipart_upload`, `generate_presigned_url`, `generate_presigned_post`

- **S3TorchConnector**: `SequentialS3Reader.read`, `S3Writer.write`

- **Tensorflow-io**: `read`, `readline`, `readlines`, `write`, `copy`, `copy_v2` via `tf.io.gfile` with `s3://` paths

For more information about S3 tracing capabilities and setup, refer to [S3 Trace](../UserGuide/index.html#s3-trace) in the User Guide.

**Prerequisites**

This recipe requires that Nsight Systems reports be collected with specific tracing parameters:

- `--trace=s3` - Enables S3 operation tracing.  
  - This recipe will also work with `--trace=s3-verbose`, but does not use any additional data from it.

- **Optional:** To enable tracing of MPI rank information, use `--trace=mpi` along with either `--mpi-impl=openmpi` or `--mpi-impl=mpich`.

- **Optional:** To enable the NVTX range correlation table, instrument your application code with NVTX ranges. See [Marking and Labeling Regions](../UserGuide/index.html#marking-and-labeling-regions) in the User Guide.

**Usage**

``` text
[1] Create a reports folder.
[2] Collect nsys-rep reports, using '--trace=s3' parameter, and save them to the reports folder.
[3] Run the recipe, using 'nsys recipe s3_access_sum --input [reports folder path]'.
```

**Output**

As the main output, the recipe generates an interactive Jupyter notebook `s3_access_stats.ipynb` with the following sections:

- Bucket Access Summary:  
  

- Objects Access Summary:  
  

- Hottest Downloaded Objects:  
  

- All Objects Table:  
  

- S3 Access Statistics for NVTX Ranges:  
  

### Recipe Catalog and Sample Results

Post-Collection Analysis enables deep insights into GPU application performance through a curated set of analysis recipes. Each recipe generates specialized visualizations and data summaries from profiling data collected during application execution.

This catalog documents all available recipes, their output formats, and sample results from profiling Llama-70B inference on 8 NVIDIA H100 GPUs using the NVIDIA Dynamo framework.

The recipes are organized by functionality:

- CUDA API Recipes: Analyze CUDA kernel launches, memory operations, and synchronization.

- GPU Kernel Recipes: Examine kernel execution times, duration distribution, and pacing.

- NVTX Recipes: Analyze application-level instrumentation ranges.

- Communication Recipes: Study multi-GPU communication patterns, including NCCL and NVLink.

- System Recipes: Review OS runtime and I/O operations.

- Comparison Recipes: Compare results between different profiling runs.

Each recipe includes sample data tables and visualization mockups representative of real profiling results.

#### CUDA API Recipes

**CUDA API Summary**

Summarizes CUDA API calls by frequency and execution time. Identifies the most time-consuming CUDA operations.

**Output Format:** Summary table, top items bar chart, and duration distribution box plot.

| API Function          | Call Count | Total Time (us) | Avg Time (us) | Max Time (us) |
|-----------------------|------------|-----------------|---------------|---------------|
| cudaMemcpy            | 2850       | 125400.5        | 43.98         | 650.2         |
| cudaLaunchKernel      | 5600       | 85300.2         | 15.23         | 420.5         |
| cudaDeviceSynchronize | 280        | 34520.0         | 123.29        | 850.3         |
| cudaMemset            | 450        | 12350.8         | 27.45         | 180.1         |
| cudaMalloc            | 125        | 8900.3          | 71.20         | 340.2         |

[](../_images/page_04_cuda_api_summary.png)

**CUDA Synchronization APIs**

Advisory analysis identifying synchronization bottlenecks and suggesting optimization strategies.

**Output Format:** Advisory table with severity levels and recommendations.

| Issue | Severity | Count | Total Impact (us) | Recommendation |
|----|----|----|----|----|
| Implicit Synchronization | HIGH | 45 | 28500 | Use async transfers with events |
| Unnecessary cudaDeviceSynchronize | MEDIUM | 28 | 12300 | Replace with stream events |
| Blocking Memory Transfers | MEDIUM | 156 | 45200 | Use pinned memory for H2D |

##### CUDA GPU Kernel Recipes

**CUDA GPU Kernel Summary**

Comprehensive summary of GPU kernels executed. Shows kernel names, execution counts, and timing statistics.

**Output Format:** Summary table, top kernels bar chart, and duration distribution box plot.

| Kernel Name | Launch Count | Total Time (us) | Avg Time (us) | Max Time (us) |
|----|----|----|----|----|
| volta_sgemm_128x64_nn | 8400 | 2845320.0 | 338.73 | 4250.2 |
| attention_forward_kernel | 2800 | 1950200.5 | 696.50 | 2850.3 |
| layernorm_forward_kernel | 2800 | 892340.2 | 318.69 | 1620.5 |
| fused_mlp_kernel | 2800 | 756200.1 | 270.07 | 1450.2 |
| nccl_AllReduce_Ring_LL_Sum_f16 | 560 | 680450.3 | 1214.20 | 3200.1 |

[](../_images/page_06_cuda_gpu_kernel_summary.png)

**CUDA GPU Kernel Duration Histogram**

Distribution of kernel execution durations. Shows frequency of kernels across different execution time ranges.

**Output Format:** Histogram with distribution statistics and percentile markers.

| Duration Range (us) | Frequency | Percentage | Cumulative % |
|---------------------|-----------|------------|--------------|
| 100-500             | 1240      | 8.2%       | 8.2%         |
| 500-1000            | 2850      | 18.7%      | 26.9%        |
| 1000-2000           | 4200      | 27.5%      | 54.4%        |
| 2000-4000           | 3950      | 25.9%      | 80.3%        |
| 4000+               | 2860      | 18.7%      | 99.0%        |

[](../_images/page_07_cuda_gpu_kernel_duration_histogram.png)

**CUDA GPU Kernel Pacing**

Consistency analysis of kernel execution times across iterations and GPU ranks. Identifies pacing irregularities.

**Output Format:** Multi-line chart showing kernel timing per iteration across all GPU ranks.

| Iteration | Rank 0 (us) | Rank 1 (us) | Rank 2 (us) | Rank 3 (us) | Avg Variance |
|-----------|-------------|-------------|-------------|-------------|--------------|
| 0         | 1025.3      | 1048.2      | 1032.5      | 1041.8      | 8.6          |
| 1         | 1032.8      | 1055.3      | 1038.2      | 1049.5      | 8.9          |
| 2         | 1038.5      | 1062.1      | 1044.8      | 1055.2      | 9.2          |
| 3         | 1035.2      | 1058.9      | 1040.5      | 1051.8      | 8.7          |

[](../_images/page_08_cuda_gpu_kernel_pacing.png)

##### CUDA Memory Operations Recipes

**CUDA GPU MemOps Summary (by Size)**

Memory transfer operations grouped by transfer size. Analyzes efficiency of data movement.

**Output Format:** Summary table, size distribution bar chart, and timing box plot.

| Size Range (MB) | Transfer Count | Total Data (GB) | Avg Time (us) | Bandwidth (GB/s) |
|----|----|----|----|----|
| \<1 | 8450 | 5.2 | 125.3 | 41.5 |
| 1-10 | 3200 | 18.5 | 450.2 | 41.0 |
| 10-100 | 850 | 42.3 | 1850.5 | 22.8 |
| 100-1000 | 280 | 156.8 | 8520.3 | 18.4 |
| 1000+ | 45 | 234.5 | 25300.2 | 9.3 |

[](../_images/page_09_cuda_gpu_memops_summary_by_size.png)

**CUDA GPU MemOps Summary (by Time)**

Memory operations sorted by execution time. Identifies most time-consuming memory transfers.

**Output Format:** Summary table, top operations bar chart, and duration distribution box plot.

| Operation Type | Count | Total Time (us) | Avg Time (us) | Max Time (us) |
|----------------|-------|-----------------|---------------|---------------|
| H2D Copy       | 4200  | 285400.2        | 67.95         | 850.3         |
| D2H Copy       | 3850  | 256300.5        | 66.57         | 720.2         |
| D2D Copy       | 2950  | 156800.3        | 53.15         | 580.1         |
| Memset         | 450   | 12350.8         | 27.45         | 180.5         |

[](../_images/page_10_cuda_gpu_memops_summary_by_time.png)

**CUDA Synchronous Memcpy**

Advisory for synchronous memory copies that may impact performance.

**Output Format:** Advisory table with impact analysis and optimization suggestions.

| Transfer Pattern | Count | Total Time (us) | Severity | Suggestion                   |
|------------------|-------|-----------------|----------|------------------------------|
| Pageable H2D     | 1250  | 145200          | HIGH     | Use cuda-pinned memory       |
| Pageable D2H     | 1100  | 128500          | HIGH     | Implement async with streams |
| Device Sync      | 280   | 34500           | MEDIUM   | Use cudaEventSynchronize     |

##### GPU Utilization Recipes

**GPU Metrics Utilization Summary**

High-level metrics showing GPU resource utilization, including SM, Memory, Tensor Cores, and related metrics.

**Output Format:** Summary table, metric utilization bar chart, and distribution box plot.

| Metric             | Avg Util (%) | Max Util (%) | Peak Time (ms) | Stalls |
|--------------------|--------------|--------------|----------------|--------|
| SM Utilization     | 78.5         | 95.2         | 245.3          | 12     |
| Memory Utilization | 65.3         | 88.1         | 238.5          | 28     |
| L1 Cache Hit Rate  | 82.4         | 96.1         | 250.2          | 5      |
| L2 Cache Hit Rate  | 74.2         | 91.3         | 242.8          | 18     |
| Tensor Utilization | 81.5         | 94.7         | 246.1          | 8      |

[](../_images/page_12_gpu_metrics_utilization_summary.png)

**GPU Metric Utilization Heatmap**

Time-series heatmap showing metric utilization across GPU ranks over profiling duration.

**Output Format:** 2D heatmap with GPU ranks on the y-axis, time bins on the x-axis, and utilization percentage color-coded.

| GPU Rank | T0-T100 (%) | T100-T200 (%) | T200-T300 (%) | T300-T400 (%) | Avg (%) |
|----------|-------------|---------------|---------------|---------------|---------|
| Rank 0   | 82          | 85            | 79            | 84            | 82.5    |
| Rank 1   | 80          | 83            | 78            | 82            | 80.8    |
| Rank 2   | 81          | 84            | 80            | 85            | 82.5    |
| Rank 3   | 79          | 82            | 77            | 81            | 79.8    |

[](../_images/page_13_gpu_metric_utilization_heatmap.png)

**Graphics VRAM Usage**

Memory allocation and deallocation trace showing VRAM consumption over time.

**Output Format:** Time-series line chart with peak VRAM markers and allocation events.

| Time (ms) | Allocated (MB) | Free (MB) | Total GPU Mem (MB) | Peak Usage (%) |
|-----------|----------------|-----------|--------------------|----------------|
| 0         | 2048           | 77952     | 80000              | 2.6%           |
| 100       | 8192           | 71808     | 80000              | 10.2%          |
| 200       | 24576          | 55424     | 80000              | 30.7%          |
| 300       | 38912          | 41088     | 80000              | 48.6%          |
| 400       | 42048          | 37952     | 80000              | 52.6%          |

[](../_images/page_14_graphics_vram_usage.png)

**GPU Time Utilization**

Advisory on GPU idle time, gaps, and utilization efficiency.

**Output Format:** Advisory table with gap analysis and improvement suggestions.

| GPU | Active Time (ms) | Idle Time (ms) | Util (%) | Largest Gap (us) | Recommendation |
|----|----|----|----|----|----|
| GPU 0 | 485.2 | 14.8 | 96.8% | 2850 | Excellent utilization |
| GPU 1 | 478.5 | 21.5 | 95.7% | 3200 | Minor gaps detected |
| GPU 2 | 475.3 | 24.7 | 95.1% | 3850 | Check kernel dependencies |
| GPU 3 | 472.8 | 27.2 | 94.6% | 4100 | Optimize host-device sync |

**GPU Gaps**

Identifies periods of GPU inactivity and analyzes root causes.

**Output Format:** Gap summary table with timeline and impact analysis.

| Gap ID  | Start (us) | Duration (us) | GPU   | Probable Cause     | Impact         |
|---------|------------|---------------|-------|--------------------|----------------|
| Gap_001 | 125340     | 2850          | GPU 0 | Host-Device Sync   | Kernel stall   |
| Gap_002 | 245820     | 3200          | GPU 1 | Memory Transfer    | Pipeline flush |
| Gap_003 | 385500     | 1950          | GPU 2 | Context Switch     | Minor impact   |
| Gap_004 | 458200     | 4100          | GPU 3 | Host Queue Latency | Kernel delay   |

##### NVTX Range Recipes

**NVTX Range Summary**

Summary of NVIDIA Tools eXtension (NVTX) instrumentation ranges in application code.

**Output Format:** Summary table, top ranges bar chart, and duration distribution box plot.

| Range Name        | Count | Total Time (us) | Avg Time (us) | Max Time (us) |
|-------------------|-------|-----------------|---------------|---------------|
| forward_pass      | 280   | 1850320.5       | 6608.29       | 8520.3        |
| attention_compute | 2800  | 950200.2        | 339.36        | 2850.1        |
| mlp_compute       | 2800  | 756300.8        | 270.11        | 1520.2        |
| backward_pass     | 280   | 1420500.3       | 5073.21       | 7250.3        |
| parameter_update  | 280   | 385200.1        | 1375.71       | 2100.5        |

[](../_images/page_17_nvtx_range_summary.png)

**NVTX Pacing**

Pacing consistency analysis of NVTX ranges across iterations.

**Output Format:** Multi-line chart showing range timing per iteration across GPU ranks.

| Iteration | Rank 0 (us) | Rank 1 (us) | Rank 2 (us) | Rank 3 (us) | Std Dev |
|-----------|-------------|-------------|-------------|-------------|---------|
| 0         | 6580.2      | 6625.8      | 6590.5      | 6612.3      | 15.8    |
| 1         | 6598.5      | 6642.1      | 6608.2      | 6630.5      | 16.2    |
| 2         | 6612.3      | 6658.9      | 6622.8      | 6645.2      | 16.8    |
| 3         | 6625.8      | 6672.5      | 6638.5      | 6660.1      | 17.3    |

[](../_images/page_18_nvtx_pacing.png)

**NVTX GPU Projection Summary**

GPU utilization projection for NVTX ranges. Shows estimated GPU time within each range.

**Output Format:** Summary table with GPU projection metrics, bar chart, and box plot.

| Range Name        | GPU Time (us) | Total Time (us) | GPU % | Kernel Calls |
|-------------------|---------------|-----------------|-------|--------------|
| forward_pass      | 1750280.3     | 1850320.5       | 94.6% | 2850         |
| attention_compute | 905200.5      | 950200.2        | 95.3% | 2800         |
| mlp_compute       | 725300.2      | 756300.8        | 95.9% | 2800         |
| backward_pass     | 1385400.1     | 1420500.3       | 97.5% | 2800         |
| parameter_update  | 370200.8      | 385200.1        | 96.1% | 280          |

[](../_images/page_19_nvtx_gpu_projection_summary.png)

**NVTX GPU Projection Pacing**

Consistency of GPU utilization for NVTX ranges across iterations and ranks.

**Output Format:** Multi-line pacing chart with GPU projection metrics.

[](../_images/page_20_nvtx_gpu_projection_pacing.png)

**NVTX GPU Projection Trace**

Detailed trace showing compute versus communication overlap within NVTX ranges.

**Output Format:** Overlap matrix and timeline visualization.

| Range             | Compute (us) | Comm (us) | Overlap (us) | Overlap % |
|-------------------|--------------|-----------|--------------|-----------|
| forward_pass      | 1650280      | 185320    | 152850       | 82.5%     |
| attention_compute | 850200       | 120200    | 98560        | 82.0%     |
| mlp_compute       | 725300       | 95300     | 78520        | 82.5%     |
| backward_pass     | 1285400      | 210500    | 172450       | 81.9%     |
| parameter_update  | 285200       | 65200     | 52850        | 81.0%     |

[](../_images/page_21_nvtx_gpu_projection_trace.png)

##### Communication & NCCL Recipes

**NCCL Summary**

Comprehensive summary of NVIDIA Collective Communications Library (NCCL) operations.

**Output Format:** Summary table, operation distribution bar chart, and timing box plot.

| Operation     | Count | Total Time (us) | Avg Time (us) | Max Time (us) |
|---------------|-------|-----------------|---------------|---------------|
| AllReduce     | 560   | 680450.3        | 1214.20       | 3200.1        |
| ReduceScatter | 560   | 520300.5        | 928.75        | 2850.2        |
| AllGather     | 560   | 485200.2        | 866.07        | 2520.3        |
| Broadcast     | 280   | 185300.8        | 661.79        | 1850.2        |
| Barrier       | 560   | 125400.1        | 223.93        | 620.5         |

[](../_images/page_23_nccl_summary.png)

**NCCL GPU Projection Summary**

GPU utilization during NCCL operations. Breaks down compute versus communication time.

**Output Format:** Summary table with GPU projection, bar chart, and box plot.

| NCCL Op       | GPU Time (us) | Total Time (us) | GPU Util % | Rank Participation |
|---------------|---------------|-----------------|------------|--------------------|
| AllReduce     | 645200.2      | 680450.3        | 94.8%      | 8                  |
| ReduceScatter | 485300.1      | 520300.5        | 93.3%      | 8                  |
| AllGather     | 450200.3      | 485200.2        | 92.8%      | 8                  |
| Broadcast     | 175300.5      | 185300.8        | 94.6%      | 8                  |
| Barrier       | 118200.2      | 125400.1        | 94.3%      | 8                  |

[](../_images/page_24_nccl_gpu_projection_summary.png)

**NCCL GPU Time Utilization Heatmap**

Time-series heatmap of GPU utilization during NCCL operations across all ranks.

**Output Format:** 2D heatmap with GPU ranks on the y-axis and time bins on the x-axis.

| GPU Rank | T0 (%) | T1 (%) | T2 (%) | T3 (%) | Avg (%) |
|----------|--------|--------|--------|--------|---------|
| Rank 0   | 92     | 94     | 93     | 95     | 93.5    |
| Rank 1   | 90     | 92     | 91     | 93     | 91.5    |
| Rank 2   | 91     | 93     | 92     | 94     | 92.5    |
| Rank 3   | 89     | 91     | 90     | 92     | 90.5    |

[](../_images/page_25_nccl_gpu_time_utilization_heatmap.png)

**NCCL GPU Overlap Trace**

Overlap analysis between communication and compute during NCCL operations.

**Output Format:** Overlap matrix showing communication overlap percentages between ranks.

| Op | Overlap % | Compute Time (us) | Comm Time (us) | Concurrent Time (us) |
|----|----|----|----|----|
| AllReduce | 78.5 | 620300 | 145200 | 113850 |
| ReduceScatter | 76.2 | 450200 | 98500 | 75000 |
| AllGather | 74.8 | 425300 | 85200 | 63800 |
| Broadcast | 72.5 | 160300 | 32500 | 23600 |
| Barrier | 65.3 | 108200 | 18500 | 12100 |

[](../_images/page_26_nccl_gpu_overlap_trace.png)

##### Network & Storage Recipes

**Network Traffic Summary**

Summary of network I/O operations and traffic patterns.

**Output Format:** Summary table, traffic distribution bar chart, and timing box plot.

| Interface  | Packets | Bytes (MB) | Avg Latency (us) | Max Latency (us) |
|------------|---------|------------|------------------|------------------|
| eth0       | 85230   | 2048.5     | 125.3            | 850.2            |
| eth1       | 75600   | 1820.3     | 130.5            | 920.3            |
| InfiniBand | 125400  | 3200.8     | 85.2             | 520.1            |

[](../_images/page_28_network_traffic_summary.png)

**Network Devices Traffic Heatmap**

Time-series heatmap showing network device utilization over profiling duration.

**Output Format:** 2D heatmap with network devices on the y-axis and time bins on the x-axis.

| Device     | T0 (Mb/s) | T1 (Mb/s) | T2 (Mb/s) | T3 (Mb/s) | Avg (Mb/s) |
|------------|-----------|-----------|-----------|-----------|------------|
| eth0       | 2500      | 2350      | 2400      | 2450      | 2425       |
| eth1       | 2200      | 2100      | 2150      | 2200      | 2162       |
| InfiniBand | 3500      | 3400      | 3450      | 3550      | 3475       |

[](../_images/page_29_network_devices_traffic_heatmap.png)

**AWS Metrics Heatmap**

CloudWatch metrics heatmap for AWS-deployed systems.

**Output Format:** Heatmap of AWS metrics, including CPU, Memory, and Network, over time.

| Metric       | T0 (%) | T1 (%) | T2 (%) | T3 (%) | Avg (%) |
|--------------|--------|--------|--------|--------|---------|
| CPU Util     | 75     | 78     | 76     | 79     | 77.0    |
| Memory Util  | 62     | 64     | 63     | 65     | 63.5    |
| Network Util | 58     | 60     | 59     | 62     | 59.8    |

[](../_images/page_30_aws_metrics_heatmap.png)

**Storage Metrics Heatmap**

Storage I/O and utilization metrics over profiling duration.

**Output Format:** Heatmap showing storage device utilization percentages.

| Storage Device | T0 (%) | T1 (%) | T2 (%) | T3 (%) | Avg (%) |
|----------------|--------|--------|--------|--------|---------|
| NVMe-0         | 45     | 48     | 46     | 50     | 47.25   |
| NVMe-1         | 42     | 44     | 43     | 46     | 43.75   |
| HDD-0          | 28     | 30     | 29     | 32     | 29.75   |

[](../_images/page_31_storage_metrics_heatmap.png)

##### System & OS Runtime Recipes

**OS Runtime Summary**

Summary of OS runtime operations including system calls and I/O operations.

**Output Format:** Summary table, operation distribution bar chart, and timing box plot.

| OS Operation | Call Count | Total Time (us) | Avg Time (us) | Max Time (us) |
|--------------|------------|-----------------|---------------|---------------|
| read()       | 1850       | 28500.3         | 15.41         | 250.2         |
| write()      | 1620       | 24300.5         | 15.01         | 220.3         |
| open()       | 450        | 8500.2          | 18.89         | 120.5         |
| close()      | 425        | 7200.1          | 16.94         | 95.2          |
| mmap()       | 120        | 4200.8          | 35.01         | 180.3         |

[](../_images/page_32_os_runtime_summary.png)

**Linux File Access Summary**

Specialized analysis of file system access patterns during profiling.

**Output Format:** File access statistics table with I/O patterns.

| File/Path       | Access Count | Bytes Read | Bytes Written | Total Time (us) |
|-----------------|--------------|------------|---------------|-----------------|
| /data/model.bin | 250          | 2048000    | 0             | 18500.2         |
| /tmp/cache      | 1850         | 512000     | 256000        | 12300.5         |
| /proc/stat      | 450          | 0          | 0             | 2800.3          |
| /dev/null       | 600          | 0          | 1024000       | 8200.1          |

[](../_images/file_access_files_summary.png)

##### NVLink & MPI Recipes

**NVLink Network Throughput Summary**

Summary of NVLink GPU-to-GPU communication throughput and efficiency.

**Output Format:** Summary table, throughput distribution bar chart, and timing box plot.

| Link Pair | Transfers | Total Data (MB) | Total Time (us) | Throughput (GB/s) |
|-----------|-----------|-----------------|-----------------|-------------------|
| GPU0-GPU1 | 450       | 2048.5          | 2350.3          | 871.0             |
| GPU1-GPU2 | 450       | 2048.3          | 2340.2          | 875.1             |
| GPU2-GPU3 | 450       | 2048.1          | 2360.5          | 868.2             |
| GPU4-GPU5 | 450       | 2048.2          | 2345.8          | 873.1             |
| GPU6-GPU7 | 450       | 2048.0          | 2355.2          | 869.8             |

[](../_images/page_34_nvlink_network_throughput_summary.png)

**MPI Summary**

Summary of Message Passing Interface (MPI) operations and communication patterns.

**Output Format:** Summary table, operation distribution bar chart, and timing box plot.

| MPI Operation | Call Count | Total Data (MB) | Total Time (us) | Avg Time (us) |
|---------------|------------|-----------------|-----------------|---------------|
| MPI_Allreduce | 560        | 520.0           | 680450.3        | 1214.20       |
| MPI_Reduce    | 280        | 280.5           | 350200.2        | 1250.71       |
| MPI_Bcast     | 280        | 256.2           | 185300.5        | 661.79        |
| MPI_Send      | 1850       | 850.3           | 125400.2        | 67.81         |
| MPI_Recv      | 1850       | 850.3           | 128200.1        | 69.30         |

[](../_images/page_35_mpi_summary.png)

**MPI and GPU Time Utilization Heatmap**

Correlation heatmap showing GPU utilization during MPI operations across ranks.

**Output Format:** 2D heatmap with MPI ranks on the y-axis and time bins on the x-axis.

| MPI Rank | T0 (%) | T1 (%) | T2 (%) | T3 (%) | Avg (%) |
|----------|--------|--------|--------|--------|---------|
| Rank 0   | 88     | 90     | 89     | 92     | 89.75   |
| Rank 1   | 86     | 88     | 87     | 90     | 87.75   |
| Rank 2   | 87     | 89     | 88     | 91     | 88.75   |
| Rank 3   | 85     | 87     | 86     | 89     | 86.75   |

[](../_images/page_36_mpi_and_gpu_time_utilization_heatmap.png)

**UCX and GPU Time Utilization Heatmap**

Heatmap showing GPU utilization during UCX (Unified Communication X) operations.

**Output Format:** 2D heatmap with UCX operation types on the y-axis and time bins on the x-axis.

| UCX Op Type | T0 (%) | T1 (%) | T2 (%) | T3 (%) | Avg (%) |
|-------------|--------|--------|--------|--------|---------|
| UCX_Put     | 82     | 84     | 83     | 86     | 83.75   |
| UCX_Get     | 79     | 81     | 80     | 83     | 80.75   |
| UCX_Atomic  | 75     | 77     | 76     | 79     | 76.75   |

[](../_images/page_37_ucx_and_gpu_time_utilization_heatmap.png)

##### Comparison & Diff Recipes

**Statistics Diff**

Side-by-side comparison of profiling statistics from two different runs.

**Output Format:** Comparison table with Run A versus Run B metrics and percentage differences.

| Metric                  | Run A  | Run B  | Difference | Change % |
|-------------------------|--------|--------|------------|----------|
| Total GPU Time (s)      | 485.2  | 492.5  | +7.3       | +1.5%    |
| Avg Kernel Time (us)    | 338.73 | 341.25 | +2.52      | +0.7%    |
| Memory Bandwidth (GB/s) | 750.2  | 745.8  | -4.4       | -0.6%    |
| NCCL Overhead (ms)      | 125.3  | 128.5  | +3.2       | +2.6%    |
| Peak VRAM (MB)          | 42048  | 42512  | +464       | +1.1%    |

[](../_images/page_38_statistics_diff.png)

### Recipe Composite Tables

When writing a recipe, you use `CompositeTable` enum values to request pre-processed DataFrames. These composite tables join, rename, and resolve raw parquet export tables into analysis-ready DataFrames, so you don’t have to do the merging and ID-to-string resolution yourself.

#### Available Composite Tables

| CompositeTable | Description |
|----|----|
| `CUDA_GPU` | All GPU activities (kernels, memcpy, memset, mem_decompress) concatenated into one table. Columns are the union of the raw CUPTI activity columns. Use this when you want a broad view of everything the GPU did. |
| `CUDA_GPU_GRAPH` | Same as `CUDA_GPU` but includes `graphNodeId` and `graphId` columns for correlating activities to CUDA graph nodes. Use this when analyzing CUDA graph execution. |
| `CUDA_COMBINED` | GPU activities + CPU runtime API calls, correlated by `correlationId`. GPU timing is renamed to `gpu_start`/`gpu_end` to avoid conflicts with the runtime `start`/`end`. The runtime `nameId` is resolved to a human-readable `name` column (e.g., `"cudaLaunchKernel"`). Use this when you need to analyze the full CPU-to-GPU execution pipeline. |
| `CUDA_COMBINED_KERNEL` | Runtime + kernels only (no memcpy/memset). Like `CUDA_COMBINED` but limited to kernel launches, and kernel name columns (`shortName`, `mangledName`, `demangledName`) are resolved from string IDs to actual names. GPU timing renamed to `gpu_start`/`gpu_end`. |
| `CUDA_KERNEL` | Kernels only with `shortName`, `mangledName`, and `demangledName` resolved from integer IDs to actual string names. Use this when you only need kernel data without the runtime API correlation. |
| `NVTX` | NVTX ranges with `textId` resolved into a unified `text` column and `domainId` resolved into `domainName`. The raw `NVTX_EVENTS` table stores text as integer IDs; this table has readable strings. |
| `NCCL` | NCCL communication operations parsed from NVTX events. Includes `jsonText` fields for advanced NCCL tracing data. |
| `NCCL_API` | NCCL API-level view showing call durations, parsed from NVTX events. |
| `NCCL_GPU_OPERATIONS` | NCCL GPU-level operations view parsed from NVTX events. |
| `NIC` | Network interface metrics joined with device info. Columns include `nic_name`, `metric_name`, `GUID`, `value`, and timing. |
| `IB_SWITCH` | InfiniBand switch port metrics with `globalId` renamed to `GUID` and metric names resolved into `metric_name`. |
| `MPI` | All MPI event tables (P2P, collectives, start/wait, other) concatenated and sorted by start time. `textId` resolved to `text`. |
| `UCX` | All UCX event tables (submit, progress, general) concatenated and sorted by start time. `textId` resolved to `text`. |
| `GPU_METRICS` | GPU hardware metrics pivoted so each metric name becomes its own column (e.g., `"SMs`` ``Active"`, `"Tensor`` ``Active"`). Indexed by `timestamp` and `typeId` (GPU device). The raw `GPU_METRICS` table has one row per metric per timestamp; this table has one row per timestamp with metrics as columns. |
| `PERF_EVENTS` | CPU performance counter events enriched with `componentType` (Core/Cache/Socket), `cpu` number, and event `name`. |
| `GENERIC` | Generic metric events with type and field IDs resolved to `metricName` and `dataSrc` strings. Numeric values coalesced from int/uint/float/double into a single `metricValue` column. |

#### Choosing the Right CUDA Table

The CUDA-related composite tables can be confusing since there are five of them. Here is when to use each one:

| Table | Includes kernels | Includes memcpy/memset | Includes runtime API | When to use |
|----|----|----|----|----|
| `CUDA_GPU` | Yes | Yes | No | Overview of all GPU activity |
| `CUDA_GPU_GRAPH` | Yes | Yes | No | Same, but with CUDA graph correlation |
| `CUDA_KERNEL` | Yes | No | No | Kernel-only analysis with resolved names |
| `CUDA_COMBINED` | Yes | Yes | Yes | Full CPU-to-GPU pipeline analysis |
| `CUDA_COMBINED_KERNEL` | Yes | No | Yes | Kernel launch overhead analysis with resolved names |

#### Output Columns

Each composite table produces a DataFrame with specific columns. The columns depend on which raw tables were joined and what transformations were applied. Also note: when tables are loaded through `DataService`, bit-field decomposition may add derived columns such as `pid`, `tid`, and `gpuId`.

**CUDA_GPU**

``` text
correlationId   start   end   globalPid   deviceId   contextId   greenContextId   streamId   pid
```

**CUDA_GPU_GRAPH**

``` text
correlationId   start   end   globalPid   deviceId   contextId   greenContextId   streamId   graphNodeId   graphId   pid
```

`graphNodeId` appears on kernel/memcpy/memset rows, and `graphId` appears on graph-trace rows.

**CUDA_KERNEL**

``` text
correlationId   globalPid   start   end   deviceId   shortName   mangledName   demangledName   pid
```

The `*Name` columns contain resolved strings (not integer IDs).

**CUDA_COMBINED**

``` text
start   end   globalTid   name   correlationId   gpu_start   gpu_end   pid   tid   deviceId   contextId   greenContextId   streamId
```

`start`/`end` are the CPU-side runtime API call times. `gpu_start`/`gpu_end` are the GPU-side activity times. `name` is the resolved API function name (e.g., `"cudaLaunchKernel"`). `pid` and `tid` are extracted from `globalTid` by bit field decomposition (`globalPid` is dropped).

**CUDA_COMBINED_KERNEL**

``` text
start   end   globalTid   correlationId   gpu_start   gpu_end   pid   tid   deviceId   shortName   mangledName   demangledName
```

Like `CUDA_COMBINED` but kernel-only, with resolved kernel names instead of runtime API name. `pid` and `tid` are extracted from `globalTid` (`globalPid` is dropped).

**NVTX**

``` text
text   start   end   globalTid   endGlobalTid   domainId   domainName   eventType   pid   tid
```

`text` is the resolved NVTX annotation string. `domainName` is `"Default"` if no domain was specified.

**NCCL**

``` text
text   start   end   globalTid   endGlobalTid   domainId   eventType   jsonText   pid   tid
```

NCCL rows are filtered from NVTX to the domain named `"NCCL"` and keep NVTX timing/thread columns. `jsonText` is present when advanced NCCL tracing payloads exist.

**NCCL_API**

``` text
text   start   end   globalTid   endGlobalTid   domainId   eventType   jsonText   count   datatype   rank   pid   tid
```

Extends `NCCL` with parsed API payload fields (`count`, `datatype`, `rank`).

**NCCL_GPU_OPERATIONS**

``` text
local_id (index)   text   start   end   commRank   count   datatype
```

One row per merged GPU operation (grouped from per-channel events). `local_id` is the DataFrame index after grouping.

**MPI / UCX**

``` text
globalTid   start   end   text   pid   tid
```

All sub-tables concatenated, sorted by `start`. `text` is the resolved operation name (e.g., `"MPI_Send"`, `"MPI_Allreduce"`).

**NIC**

``` text
start   end   globalId   nicId   value   metricsListId   metricsIdx   GUID   nic_name   metric_name
```

**IB_SWITCH**

``` text
start   end   GUID   value   metricsListId   metricsIdx   metric_name
```

**GPU_METRICS**

``` text
timestamp   typeId   gpuId   SMs Active   SM Issue   Tensor Active   Unallocated Warps in Active SMs   ...
```

Each metric is a column. The available columns depend on the GPU and capture settings.

**PERF_EVENTS**

``` text
start   end   vmId   eventId   count   componentType   cpu   name
```

**GENERIC**

``` text
timestamp   typeId   genericEventId   metricName   metricValue   dataSrc   gpuId
```

#### Examples in Existing Recipes

These recipes demonstrate how composite tables are used in practice:

- `cuda_gpu_kern_sum`: Uses `CUDA_KERNEL` for kernel summary statistics

- `nvtx_sum`: Uses `NVTX` for NVTX range analysis

- `nvtx_gpu_proj_sum`: Uses `CUDA_GPU_GRAPH` + `NVTX` for GPU projection

- `network_map_aws`: Uses `GENERIC` for AWS EFA metrics

See also [Tutorial: Create a User-Defined Recipe](#create-user-defined-recipe) for a step-by-step tutorial on creating your own recipe.

### Tutorial: Create a User-Defined Recipe

The Nsight Systems recipe system is designed to be extensible and we hope that many users will use it to create their own recipes. This short tutorial will highlight the steps needed to create a recipe that is a customized version of one of the recipes that is included in the Nsight Systems recipe package.

Before starting, you may want to review [Recipe Composite Tables](#recipe-composite-tables) for an overview of the pre-processed data tables available to recipes.

**Step 1: Create the recipe directory and script**

Make a new directory in the `<install-dir>/target-linux-x64/python/packages/nsys_recipe/recipes` folder based on the name of your new recipe. For this example, we will call our new recipe new_metric_util_map. We will copy the existing gpu_metric_util_map.py script and create a new script called new_metric_util_map.py in the new_metric_util_map directory. We will also copy the heatmap.ipynb and metadata json files into the new_metric_util_map directory. Type these steps in a Linux terminal window:

``` text
> cd <install-dir>/target-linux-x64/python/packages/nsys_recipe
> mkdir new_metric_util_map
> cp gpu_metric_util_map/metadata.json new_metric_util_map/metadata.json
> cp gpu_metric_util_map/heatmap.ipynb new_metric_util_map/heatmap.ipynb
> cp gpu_metric_util_map/gpu_metric_util_map.py new_metric_util_map/new_metric_util_map.py
```

Replace the module name in `metadata.json` with new_metric_util_map and update the display name and description to your preference. Also, rename the class name `GpuMetricUtilMap` in `new_metric_util_map.py` to `NewMetricUtilMap`. We will discuss the detailed functionality of the new recipe code in the subsequent steps.

**Step 2: Modify the mapper function**

Many recipes are structured as a map-reduce algorithm. The mapper function is called for every .nsys-rep file in the report directory. The mapper function performs a series of calculations on the events in each Nsight Systems report and produces an intermediate data set. The intermediate results are then combined by the reduce function to produce the final results. The mapper function can be called in parallel, either on multiple cores of a single node (using the concurrent python module), or multiple ranks of a multi-node recipe analysis (using the Dask distributed module).

When we create a new recipe, we need to create a class that derives from the Recipe base class. For our example, that class will be called NewMetricUtilMap (which we had renamed in step 1).

The mapper function is called mapper_func(). It will first convert the .nsys-rep file into a data storage file (SQLite/Parquet/Arrow), if the file does not already exist. It then reads all the necessary tables from the exported file into Pandas Dataframes needed by the recipe. GPU Metric data is stored using a database schema table called `GENERIC_EVENTS`. For extra flexibility, `GENERIC_EVENTS` represents the data as a JSON object, which is stored as a string. The `NewMetricUtilMap` class extracts fields from the JSON object and accumulates them over the histogram bins of the heat map.

The original script retrieved three GPU metrics: SM Active, SM Issue, and Tensor Active. In our new version of the script, we will extract a fourth metric, Unallocated Warps in Active SMs.

1.  Find this line (approximately line 44):

    ``` text
    metric_cols = ["SMs Active", "SM Issue", "Tensor Active"]
    ```

2.  Add the Unallocated Warps in Active SMs metric:

    ``` text
    metric_cols = [
        "SMs Active",
        "SM Issue",
        "Tensor Active",
        "Unallocated Warps in Active SMs",
    ]
    ```

**Step 3: Modify the reduce function**

Our new mapper function will extract four GPU metrics and return them as a Pandas DataFrame. The reduce function receives a list of DataFrames, one for each .nsys-rep file in the analysis, and combines them into a single DataFrame using the Pandas concat function. Since the reducer function is generic in our case, no modifications are needed. However, if you would like to add any additional post-processing, you can do so in this function.

**Step 4: Add a plot to the Jupyter notebook**

Our new recipe class will create a Parquet output file with all the data produced by the reducer function, using the `to_parquet()` function. It will also create a Jupyter notebook file using the `create_notebook()` function.

In this step, we will change the `create_notebook()` function to produce a plot for our fourth metric. To do this, we need to change these two lines (located in the second cell of `new_metric_util_map/heatmap.ipynb`):

``` text
metrics = [
   "SMs Active",
   "SM Issue",
   "Tensor Active",
]
```

To this:

``` text
metrics = [
    "SMs Active",
    "SM Issue",
    "Tensor Active",
    "Unallocated Warps in Active SMs",
]
```

That completes all the modifications for our NewMetricUtilMap class.

**Step 5: Run the new recipe**

If the new recipe is located in the default recipe directory nsys_recipe/recipes, we can directly run it using the `nsys`` ``recipe` command like this:

``` text
> nsys recipe new_metric_util_map --input <directory of reports>
```

It is also possible to have a recipe located outside of this directory. In this case, you need to set the environment variable `NSYS_RECIPE_PATH` to the directory containing the recipe when running the `nsys`` ``recipe` command.

When successful, the recipe should produce a new recipe result directory called `new_metric_util_map-1`.

If we open the Jupyter notebook in that recipe and execute the code, we should see our new heatmap along with the three plots produced by the original version of the recipe. Here is an example:



## Available Export Formats

You may want to create your own analysis system interrogating the data produced by Nsight Systems. However, the .nsys-rep output format of Nsight Systems is not designed to be accessed by users, and we cannot guarantee its stability. Therefore you will want to export into a good format for your further processing.

### SQLite Schema Reference

Nsight Systems has the ability to export SQLite database files from the .nsys-rep results file. From the CLI, use `nsys`` ``export`. From the GUI, call `File->Export...`.

Note

The .nsys-rep report format is the only data format for Nsight Systems that should be considered forward-compatible. The SQLite schema can and will change in the future.

The schema for a concrete database can be obtained with the sqlite3 tool built-in command `.schema`. The sqlite3 tool can be located in the Target or Host directory of your Nsight Systems installation.

Note

Currently, tables are created lazily, and therefore not every table described in the documentation will be present in a particular database. This will change in a future version of the product. If you want a full schema of all possible tables, use `nsys`` ``export`` ``--lazy=false` during the export phase.

Currently, a table is created for each data type in the exported database. Since usage patterns for exported data may vary greatly and no default use cases have been established, no indexes or extra constraints are created. Instead, refer to the SQLite Examples section for a list of common recipes. This may change in a future version of the product.

To check the version of your exported SQLite file, check the value of `EXPORT_SCHEMA_VERSION` in the `META_DATA_EXPORT` table. The schema version is a common three-value major/minor/micro version number. The first value, or major value, indicates the overall format of the database, and is only changed if there is a major re-write or re-factor of the entire database format. It is assumed that if the major version changes, all scripts or queries will break. The middle, or minor, version is changed anytime there is a more localized, but potentially breaking change, such as renaming an existing column, or changing the type of an existing column. The last, or micro version is changed any time there are additions, such as a new table or column, that should not introduce any breaking change when used with well-written, best-practices queries.

The changes between schema versions are documented in `<install_dir>/host*/exporter/export_schema_version_notes.txt`.

This is the schema as of the 2026.2 release, schema version 3.25.0.

``` sql
CREATE TABLE StringIds (
    -- Consolidation of repetitive string values.

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
    value                       TEXT      NOT NULL                     -- String value.
);
CREATE TABLE ANALYSIS_FILE (
    -- Analysis file content

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
    filename                    TEXT,                                  -- File path
    contentId                   INTEGER,                               -- REFERENCES StringIds(id) -- File content
    globalPid                   INTEGER   NOT NULL                     -- Serialized GlobalId.
);
CREATE TABLE ThreadNames (
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Thread name
    priority                    INTEGER,                               -- Priority of the thread.
    globalTid                   INTEGER                                -- Serialized GlobalId.
);
CREATE TABLE ProcessStreams (
    globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    filenameId                  INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- File name
    contentId                   INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Stream content
);
CREATE TABLE TARGET_INFO_SYSTEM_ENV (
    globalVid                   INTEGER,                               -- Serialized GlobalId.
    devStateName                TEXT      NOT NULL,                    -- Device state name.
    name                        TEXT      NOT NULL,                    -- Property name.
    nameEnum                    INTEGER   NOT NULL,                    -- Property enum value.
    value                       TEXT      NOT NULL                     -- Property value.
);
CREATE TABLE TARGET_INFO_NIC_INFO (
    GUID                        INTEGER   NOT NULL,                    -- Network interface GUID
    stateName                   TEXT      NOT NULL,                    -- Device state name
    nicId                       INTEGER   NOT NULL,                    -- Network interface Id
    name                        TEXT      NOT NULL,                    -- Network interface name
    deviceId                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_NET_DEVICE_ID(id)
    vendorId                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_NET_VENDOR_ID(id)
    linkLayer                   INTEGER   NOT NULL                     -- REFERENCES ENUM_NET_LINK_TYPE(id)
);
CREATE TABLE NIC_ID_MAP (
    -- Map between NIC info nicId and NIC metric globalId

    nicId                       INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NIC_INFO(nicId)
    globalId                    INTEGER   NOT NULL                     -- REFERENCES NET_NIC_METRIC(globalId)
);
CREATE TABLE TARGET_INFO_SESSION_START_TIME (
    utcEpochNs                  INTEGER,                               -- UTC Epoch timestamp at start of the capture (ns).
    utcTime                     TEXT,                                  -- Start of the capture in UTC.
    localTime                   TEXT,                                  -- Start of the capture in local time of target.
    systemClockNs               INTEGER                                -- Target system clock timestamp at start of the capture (ns).
);
CREATE TABLE ANALYSIS_DETAILS (
    -- Details about the analysis session.

    globalVid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    duration                    INTEGER   NOT NULL,                    -- The total time span of the entire trace (ns).
    startTime                   INTEGER   NOT NULL,                    -- Trace start timestamp in nanoseconds.
    stopTime                    INTEGER   NOT NULL                     -- Trace stop timestamp in nanoseconds.
);
CREATE TABLE PMU_EVENT_REQUESTS (
    -- PMU event requests

    id                          INTEGER   NOT NULL,                    -- PMU event request.
    eventid                     INTEGER,                               -- PMU counter event id.
    source                      INTEGER   NOT NULL,                    -- REFERENCES ENUM_PMU_EVENT_SOURCE(id)
    unit_type                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_PMU_UNIT_TYPE(id)
    event_name                  TEXT,                                  -- PMU counter unique name

    PRIMARY KEY (id)
);
CREATE TABLE TARGET_INFO_GPU (
    vmId                        INTEGER   NOT NULL,                    -- Serialized GlobalId.
    id                          INTEGER   NOT NULL,                    -- Device ID.
    name                        TEXT,                                  -- Device name.
    busLocation                 TEXT,                                  -- PCI bus location.
    isDiscrete                  INTEGER,                               -- True if discrete, false if integrated.
    l2CacheSize                 INTEGER,                               -- Size of L2 cache (B).
    totalMemory                 INTEGER,                               -- Total amount of memory on the device (B).
    memoryBandwidth             INTEGER,                               -- Amount of memory transferred (B).
    clockRate                   INTEGER,                               -- Clock frequency (Hz).
    smCount                     INTEGER,                               -- Number of multiprocessors on the device.
    pwGpuId                     INTEGER,                               -- PerfWorks GPU ID.
    uuid                        TEXT,                                  -- Device UUID.
    luid                        INTEGER,                               -- Device LUID.
    chipName                    TEXT,                                  -- Chip name.
    cuDevice                    INTEGER,                               -- CUDA device ID.
    ctxswDevPath                TEXT,                                  -- GPU context switch device node path.
    ctrlDevPath                 TEXT,                                  -- GPU control device node path.
    revision                    INTEGER,                               -- Revision number.
    nodeMask                    INTEGER,                               -- Device node mask.
    constantMemory              INTEGER,                               -- Memory available on device for __constant__ variables (B).
    maxIPC                      INTEGER,                               -- Maximum instructions per count.
    maxRegistersPerBlock        INTEGER,                               -- Maximum number of 32-bit registers available per block.
    maxShmemPerBlock            INTEGER,                               -- Maximum optin shared memory per block.
    maxShmemPerBlockOptin       INTEGER,                               -- Maximum optin shared memory per block.
    maxShmemPerSm               INTEGER,                               -- Maximum shared memory available per multiprocessor (B).
    maxRegistersPerSm           INTEGER,                               -- Maximum number of 32-bit registers available per multiprocessor.
    threadsPerWarp              INTEGER,                               -- Warp size in threads.
    asyncEngines                INTEGER,                               -- Number of asynchronous engines.
    maxWarpsPerSm               INTEGER,                               -- Maximum number of warps per multiprocessor.
    maxBlocksPerSm              INTEGER,                               -- Maximum number of blocks per multiprocessor.
    maxThreadsPerBlock          INTEGER,                               -- Maximum number of threads per block.
    maxBlockDimX                INTEGER,                               -- Maximum X-dimension of a block.
    maxBlockDimY                INTEGER,                               -- Maximum Y-dimension of a block.
    maxBlockDimZ                INTEGER,                               -- Maximum Z-dimension of a block.
    maxGridDimX                 INTEGER,                               -- Maximum X-dimension of a grid.
    maxGridDimY                 INTEGER,                               -- Maximum Y-dimension of a grid.
    maxGridDimZ                 INTEGER,                               -- Maximum Z-dimension of a grid.
    computeMajor                INTEGER,                               -- Major compute capability version number.
    computeMinor                INTEGER,                               -- Minor compute capability version number.
    smMajor                     INTEGER,                               -- Major multiprocessor version number.
    smMinor                     INTEGER                                -- Minor multiprocessor version number.
);
CREATE TABLE TARGET_INFO_XMC_SPEC (
    vmId                        INTEGER   NOT NULL,                    -- Serialized GlobalId.
    clientId                    INTEGER   NOT NULL,                    -- Client ID.
    type                        TEXT      NOT NULL,                    -- Client type.
    name                        TEXT      NOT NULL,                    -- Client name.
    groupId                     TEXT      NOT NULL                     -- Client group ID.
);
CREATE TABLE TARGET_INFO_CUDA_DEVICE (
    gpuId                       INTEGER,                               -- GPU ID.
    cudaId                      INTEGER   NOT NULL,                    -- CUDA device ID.
    pid                         INTEGER   NOT NULL,                    -- Process ID.
    uuid                        TEXT,                                  -- Device UUID.
    numMultiprocessors          INTEGER                                -- Number of SMs available on the device.
);
CREATE TABLE TARGET_INFO_PROCESS (
    processId                   INTEGER   NOT NULL,                    -- Process ID.
    openGlVersion               TEXT      NOT NULL,                    -- OpenGL version.
    correlationId               INTEGER   NOT NULL,                    -- Correlation ID of the kernel.
    nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
);
CREATE TABLE TARGET_INFO_NVTX_CUDA_DEVICE (
    name                        TEXT      NOT NULL,                    -- CUDA device name assigned using NVTX.
    hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    deviceId                    INTEGER   NOT NULL                     -- Device ID.
);
CREATE TABLE TARGET_INFO_NVTX_CUDA_CONTEXT (
    name                        TEXT      NOT NULL,                    -- CUDA context name assigned using NVTX.
    hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    processId                   INTEGER   NOT NULL,                    -- Process ID.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL                     -- Context ID.
);
CREATE TABLE TARGET_INFO_NVTX_CUDA_STREAM (
    name                        TEXT      NOT NULL,                    -- CUDA stream name assigned using NVTX.
    hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    processId                   INTEGER   NOT NULL,                    -- Process ID.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    streamId                    INTEGER   NOT NULL                     -- Stream ID.
);
CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
    nullStreamId                INTEGER   NOT NULL,                    -- Stream ID.
    hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    processId                   INTEGER   NOT NULL,                    -- Process ID.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    parentContextId             INTEGER,                               -- For green context, this is the parent context id.
    isGreenContext              INTEGER,                               -- Is this a Green Context?
    numMultiprocessors          INTEGER,                               -- For green context, number of SMs allocated.
    numTpcs                     INTEGER,                               -- For green context, number of TPCs allocated.
    tpcMask                     TEXT,                                  -- For green context, comma-separated hex TPC mask values (e.g., '0xf,0x0').
    workqueueResourceId         INTEGER,                               -- For green context, workqueue resource ID.
    workqueueConcurrencyLimit   INTEGER,                               -- For green context, workqueue concurrency limit.
    workqueueSharingScope       INTEGER                                -- For green context, workqueue sharing scope.
);
CREATE TABLE TARGET_INFO_CUDA_STREAM (
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    processId                   INTEGER   NOT NULL,                    -- Process ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    priority                    INTEGER   NOT NULL,                    -- Priority of the stream.
    flag                        INTEGER   NOT NULL                     -- REFERENCES ENUM_CUPTI_STREAM_TYPE(id)
);
CREATE TABLE TARGET_INFO_WDDM_CONTEXTS (
    context                     INTEGER   NOT NULL,
    engineType                  INTEGER   NOT NULL,
    nodeOrdinal                 INTEGER   NOT NULL,
    friendlyName                TEXT      NOT NULL
);
CREATE TABLE TARGET_INFO_PERF_METRIC (
    id                          INTEGER   NOT NULL,                    -- Event or Metric ID value
    name                        TEXT      NOT NULL,                    -- Event or Metric name
    description                 TEXT      NOT NULL,                    -- Event or Metric description
    unit                        TEXT      NOT NULL,                    -- Event or Metric measurement unit
    displayName                 TEXT                                   -- GUI friendly name of the Event or Metric
);
CREATE TABLE TARGET_INFO_NETWORK_METRICS (
    metricsListId               INTEGER   NOT NULL,                    -- Metric list ID
    metricsIdx                  INTEGER   NOT NULL,                    -- List index of metric
    name                        TEXT      NOT NULL,                    -- Name of metric
    description                 TEXT      NOT NULL,                    -- Description of metric
    unit                        TEXT      NOT NULL                     -- Measurement unit of metric
);
CREATE TABLE TARGET_INFO_COMPONENT (
    componentId                 INTEGER   NOT NULL,                    -- Component ID
    name                        TEXT      NOT NULL,                    -- Component name
    instance                    INTEGER,                               -- Component instance
    parentId                    INTEGER                                -- Parent Component ID
);
CREATE TABLE NET_IB_DEVICE_INFO (
    networkId                   INTEGER   NOT NULL,                    -- The Device's Network ID
    guid                        INTEGER,                               -- Device Guid
    name                        TEXT,                                  -- Device Name
    des                         TEXT,                                  -- Device description
    lid                         INTEGER                                -- Device Lid
);
CREATE TABLE NET_IB_DEVICE_PORT_INFO (
    guid                        INTEGER,                               -- REFERENCES NET_IB_DEVICE_INFO(guid) -- Device Global Identifier
    portNumber                  INTEGER   NOT NULL,                    -- Internal Port Number
    portLabel                   TEXT      NOT NULL,                    -- Port Label
    portLid                     INTEGER   NOT NULL                     -- Port Lid
);
CREATE TABLE NET_IB_DEVICE_TYPE_MAP (
    guid                        INTEGER,                               -- REFERENCES NET_IB_DEVICE_INFO(guid) -- Device Global Identifier
    deviceType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_NET_IB_DEVICE_TYPE(id)
);
CREATE TABLE META_DATA_CAPTURE (
    -- information about nsys capture parameters

    name                        TEXT      NOT NULL,                    -- Name of meta-data record
    value                       TEXT                                   -- Value of meta-data record
);
CREATE TABLE META_DATA_EXPORT (
    -- information about nsys export process

    name                        TEXT      NOT NULL,                    -- Name of meta-data record
    value                       TEXT                                   -- Value of meta-data record
);
CREATE TABLE ENUM_NSYS_EVENT_TYPE (
    -- Nsys event type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_EVENT_CLASS (
    -- Nsys event class labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_GENERIC_EVENT_SOURCE (
    -- Nsys generic event source labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_GENERIC_EVENT_GROUP (
    -- Nsys generic event group labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_GENERIC_EVENT_FIELD_TYPE (
    -- Nsys generic event field type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_PROPERTY (
    -- Nsys generic event field ETW property flag labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_TYPE (
    -- Nsys generic event field ETW type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_FLAGS (
    -- Nsys generic event field ETW map info flag labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_GPU_CTX_SWITCH (
    -- GPU context switch labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_MEMCPY_OPER (
    -- CUDA memcpy operation labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_MEM_KIND (
    -- CUDA memory kind labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_MEMPOOL_TYPE (
    -- CUDA mempool type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_MEMPOOL_OPER (
    -- CUDA mempool operation labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_DEV_MEM_EVENT_OPER (
    -- CUDA device mem event operation labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_KERNEL_LAUNCH_TYPE (
    -- CUDA kernel launch type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG (
    -- CUDA shared memory limit config labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_UNIF_MEM_MIGRATION (
    -- CUDA unified memory migration cause labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_UNIF_MEM_ACCESS_TYPE (
    -- CUDA unified memory access type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUDA_FUNC_CACHE_CONFIG (
    -- CUDA function cache config labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUPTI_STREAM_TYPE (
    -- CUPTI stream type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUPTI_SYNC_TYPE (
    -- CUPTI synchronization type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_CUPTI_OVERHEAD_TYPE (
    -- CUPTI overhead type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_STACK_UNWIND_METHOD (
    -- Stack unwind method labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_SAMPLING_THREAD_STATE (
    -- Sampling thread state labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_SCHEDULING_THREAD_BLOCK (
    -- Scheduling thread block labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENGL_DEBUG_SOURCE (
    -- OpenGL debug source labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENGL_DEBUG_TYPE (
    -- OpenGL debug type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENGL_DEBUG_SEVERITY (
    -- OpenGL debug severity labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_VULKAN_PIPELINE_CREATION_FLAGS (
    -- Vulkan pipeline creation feedback flag labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_VULKAN_HEAP_TYPE (
    -- Vulkan heap type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_VULKAN_HEAP_FLAGS (
    -- Vulkan heap flag labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_VULKAN_MEMORY_PROPERTY_FLAGS (
    -- Vulkan memory property flag labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_D3D12_HEAP_TYPE (
    -- D3D12 heap type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_D3D12_PAGE_PROPERTY (
    -- D3D12 CPU page property labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_D3D12_HEAP_FLAGS (
    -- D3D12 heap flag labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_D3D12_CMD_LIST_TYPE (
    -- D3D12 command list type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENACC_DEVICE (
    -- OpenACC device type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENACC_EVENT_KIND (
    -- OpenACC event type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_EVENT_KIND (
    -- OpenMP event kind labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_THREAD (
    -- OpenMP thread labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_DISPATCH (
    -- OpenMP dispatch labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_SYNC_REGION (
    -- OpenMP sync region labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_WORK (
    -- OpenMP work labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_MUTEX (
    -- OpenMP mutex labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_TASK_FLAG (
    -- OpenMP task flags labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OPENMP_TASK_STATUS (
    -- OpenMP task status labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NVDRIVER_EVENT_ID (
    -- NV-Driver event it labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_WDDM_PAGING_QUEUE_TYPE (
    -- WDDM paging queue type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_WDDM_PACKET_TYPE (
    -- WDDM packet type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_WDDM_ENGINE_TYPE (
    -- WDDM engine type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_WDDM_INTERRUPT_TYPE (
    -- WDDM DMA interrupt type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_WDDM_VIDMM_OP_TYPE (
    -- WDDM VidMm operation type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NET_LINK_TYPE (
    -- NIC link layer labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NET_DEVICE_ID (
    -- NIC PCIe device id labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NET_VENDOR_ID (
    -- NIC PCIe vendor id labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_ETW_MEMORY_TRANSFER_TYPE (
    -- memory transfer type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_PMU_EVENT_SOURCE (
    -- PMU event source labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_PMU_UNIT_TYPE (
    -- PMU unit type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_VIDEO_ENGINE_TYPE (
    -- Video engine type id labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_VIDEO_ENGINE_CODEC (
    -- Video engine codec labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL (
    -- Diagnostic message severity level labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_DIAGNOSTIC_SOURCE_TYPE (
    -- Diagnostic message source type labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE (
    -- Diagnostic message timestamp source lables

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NET_IB_DEVICE_TYPE (
    -- network device types

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_NET_IB_CONGESTION_EVENT_TYPE (
    -- IB Switch congestion event types

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE ENUM_OSRT_FILE_ACCESS_EVENT_TYPE (
    -- OSRT File Access event type

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
CREATE TABLE GENERIC_EVENT_SOURCES (
    -- Generic event source modules

    sourceId                    INTEGER   NOT NULL   PRIMARY KEY,      -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event source name
    timeSourceId                INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_GENERIC_EVENT_SOURCE(id)
    sourceGroupId               INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_GENERIC_EVENT_GROUP(id)
    hyperType                   TEXT,                                  -- Hypervisor Type
    hyperVersion                TEXT,                                  -- Hypervisor Version
    hyperStructPrefix           TEXT,                                  -- Hypervisor Struct Prefix
    hyperMacroPrefix            TEXT,                                  -- Hypervisor Macro Prefix
    hyperFilterFlags            INTEGER,                               -- Hypervisor Custom Filter Flags
    hyperDomain                 TEXT,                                  -- Hypervisor Domain
    data                        TEXT                                   -- JSON encoded generic event source description.
);
CREATE TABLE GENERIC_EVENT_TYPES (
    -- Generic event type/schema descriptions.

    typeId                      INTEGER   NOT NULL   PRIMARY KEY,      -- Serialized GlobalId.
    sourceId                    INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_SOURCES(sourceId)
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event type name
    hyperComment                TEXT,                                  -- Event Type Hypervisor Comment
    ftraceFormat                TEXT,                                  -- Event Type FTrace Format
    etwProviderId               INTEGER,                               -- Event Type ETW Provider Id
    etwProviderNameId           INTEGER,                               -- Event Type ETW Provider Name Id
    etwTaskId                   INTEGER,                               -- Event Type ETW Task Id
    etwTaskNameId               INTEGER,                               -- Event Type ETW Task Name Id
    etwEventId                  INTEGER,                               -- Event Type ETW Event Id
    etwVersion                  INTEGER,                               -- Event Type ETW Version
    etwGuidHigh                 INTEGER,                               -- Event Type ETW GUID high
    etwGuidLow                  INTEGER,                               -- Event Type ETW GUID low
    etwGuid                     TEXT,                                  -- ETW Provider GUID.
    data                        TEXT                                   -- JSON encoded generic event type description.
);
CREATE TABLE GENERIC_EVENT_TYPE_FIELDS (
    -- Generic event type/schema individual data field descriptions.

    typeId                      INTEGER   NOT NULL,                    -- Serialized GlobalId.
    fieldIdx                    INTEGER   NOT NULL,                    -- Index of type field
    fieldNameId                 INTEGER   NOT NULL,                    -- Name of field.
    offset                      INTEGER   NOT NULL,                    -- Field alignment offset size, in bytes.
    size                        INTEGER   NOT NULL,                    -- Field size, in bytes.
    type                        INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_GENERIC_EVENT_FIELD_TYPE(id)
    hyperTypeName               TEXT,                                  -- Event Field Hypervisor Type Name
    hyperFormat                 TEXT,                                  -- Event Field Hypervisor Format
    hyperComment                TEXT,                                  -- Event Field Hypervisor Comment
    ftracePrefix                TEXT,                                  -- Event Field FTrace Prefix
    ftraceSuffix                TEXT,                                  -- Event Field FTrace Suffix
    etwFlags                    INTEGER,                               -- REFERENCES ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_PROPERTY(id)
    etwCountFieldIndex          INTEGER,                               -- Event Field ETW Count Field Index
    etwLengthFieldIndex         INTEGER,                               -- Event Field ETW Length Field Index
    etwType                     INTEGER,                               -- REFERENCES ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_TYPE(id)
    etwMapInfoFlags             INTEGER,                               -- REFERENCES ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_FLAGS(id)
    etwOrderedFieldIndex        INTEGER                                -- Event Field ETW Ordered Field Index
);
CREATE TABLE GENERIC_EVENT_TYPE_FIELD_MAP (
    -- Generic event ENUM data.  Mostly used by ETW.

    typeId                      INTEGER   NOT NULL,                    -- Serialized GlobalId.
    fieldIdx                    INTEGER   NOT NULL,                    -- Index of type field
    enum                        INTEGER   NOT NULL,                    -- Event Field ETW Map Info enum.
    name                        TEXT      NOT NULL,                    -- Event Field ETW Map Info Name.
    nameId                      INTEGER   NOT NULL                     -- Event Field ETW Map Info Name Id.
);
CREATE TABLE GENERIC_EVENTS (
    -- Dynamic or unstructured event data.

    genericEventId              INTEGER   NOT NULL   PRIMARY KEY,      -- Id of particular generic event
    rawTimestamp                INTEGER   NOT NULL,                    -- Raw event timestamp recorded during profiling.
    timestamp                   INTEGER   NOT NULL,                    -- Event timestamp converted to the profiling session timeline.
    typeId                      INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_TYPES(typeId)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    data                        TEXT                                   -- JSON encoded event data.
);
CREATE TABLE GENERIC_EVENT_DATA (
    -- GENERIC_EVENTS data values.

    genericEventId              INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENTS(genericEventId)
    fieldIdx                    INTEGER   NOT NULL,                    -- Index of type field
    intVal                      INTEGER,                               -- Integer value, signed
    uintVal                     INTEGER,                               -- Integer value, unsigned
    floatVal                    REAL,                                  -- Floating point value, 32-bit
    doubleVal                   REAL                                   -- Floating point value, 64-bit
);
CREATE TABLE ETW_PROVIDERS (
    -- Names and identifiers of ETW providers captured in the report.

    providerId                  INTEGER   NOT NULL   PRIMARY KEY,      -- Provider ID.
    providerNameId              INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Provider name
    guid                        TEXT      NOT NULL                     -- ETW Provider GUID.
);
CREATE TABLE ETW_TASKS (
    -- Names and identifiers of ETW tasks captured in the report.

    taskNameId                  INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Task name
    taskId                      INTEGER   NOT NULL,                    -- The event task ID.
    providerId                  INTEGER   NOT NULL                     -- Provider ID.
);
CREATE TABLE ETW_EVENTS (
    -- Raw ETW events captured in the report.

    rawTimestamp                INTEGER   NOT NULL,                    -- Raw event timestamp recorded during profiling.
    timestamp                   INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    typeId                      INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_TYPES(typeId)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    opcode                      INTEGER,                               -- The event opcode.
    data                        TEXT      NOT NULL                     -- JSON encoded event data.
);
CREATE TABLE TARGET_INFO_GPU_METRICS (
    -- GPU Metrics, metric names and ids.

    typeId                      INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_TYPES(typeId)
    sourceId                    INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_SOURCES(sourceId)
    typeName                    TEXT      NOT NULL,                    -- Name of event type.
    metricId                    INTEGER   NOT NULL,                    -- Id of metric in event; not assumed to be stable.
    metricName                  TEXT      NOT NULL                     -- Definitive name of metric.
);
CREATE TABLE GPU_METRICS (
    -- GPU Metrics, events and values.

    rawTimestamp                INTEGER   NOT NULL,                    -- Raw event timestamp recorded during profiling.
    timestamp                   INTEGER   NOT NULL,                    -- Event timestamp (ns).
    typeId                      INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_GPU_METRICS(typeId) and GENERIC_EVENT_TYPES(typeId)
    metricId                    INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_GPU_METRICS(metricId)
    value                       INTEGER   NOT NULL                     -- Counter data value
);
CREATE TABLE TARGET_INFO_SOC_METRICS (
    -- SoC Metrics, metric names and ids.

    typeId                      INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_TYPES(typeId)
    sourceId                    INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_SOURCES(sourceId)
    typeName                    TEXT      NOT NULL,                    -- Name of event type.
    metricId                    INTEGER   NOT NULL,                    -- Id of metric in event; not assumed to be stable.
    metricName                  TEXT      NOT NULL                     -- Definitive name of metric.
);
CREATE TABLE SOC_METRICS (
    -- SoC Metrics, events and values.

    rawTimestamp                INTEGER   NOT NULL,                    -- Raw event timestamp recorded during profiling.
    timestamp                   INTEGER   NOT NULL,                    -- Event timestamp (ns).
    typeId                      INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_SOC_METRICS(typeId) and GENERIC_EVENT_TYPES(typeId)
    metricId                    INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_GPU_METRICS(metricId)
    value                       INTEGER   NOT NULL                     -- Counter data value
);
CREATE TABLE MPI_COMMUNICATORS (
    -- Identification of MPI communication groups.

    rank                        INTEGER,                               -- Active MPI rank
    timestamp                   INTEGER,                               -- Time of MPI communicator creation.
    commHandle                  INTEGER,                               -- MPI communicator handle.
    parentHandle                INTEGER,                               -- MPI communicator handle.
    localRank                   INTEGER,                               -- Local MPI rank in a communicator.
    size                        INTEGER,                               -- MPI communicator size.
    groupRoot                   INTEGER,                               -- Root rank (global) in MPI communicator.
    groupRootUid                INTEGER,                               -- Group root's communicator ID.
    members                     TEXT,                                  -- MPI communicator members (index is global rank).
    commUid                     INTEGER                                -- Globally unique MPI communicator ID.
);
CREATE TABLE NVTX_PAYLOAD_SCHEMAS (
    -- NVTX payload schema attributes.

    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    schemaId                    INTEGER,                               -- Identifier of the payload schema.
    name                        TEXT,                                  -- Schema name.
    type                        INTEGER,                               -- Schema type.
    flags                       INTEGER,                               -- Schema flags.
    numEntries                  INTEGER,                               -- Number of payload schema entries.
    payloadSize                 INTEGER,                               -- Size of the static payload.
    alignTo                     INTEGER                                -- Field alignment in bytes.
);
CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES (
    -- NVTX payload schema entries.

    domainId                    INTEGER   NOT NULL,                    -- User-controlled ID that can be used to group events.
    schemaId                    INTEGER   NOT NULL,                    -- Identifier of the payload schema.
    idx                         INTEGER   NOT NULL,                    -- Index of the entry in the payload schema.
    flags                       INTEGER,                               -- Payload entry flags.
    type                        INTEGER,                               -- Payload entry type.
    name                        TEXT,                                  -- Label of the payload entry.
    description                 TEXT,                                  -- Description of the payload entry.
    arrayOrUnionDetail          INTEGER,                               -- Array length (index) or selected union member.
    offset                      INTEGER                                -- Entry offset in the binary data in bytes.
);
CREATE TABLE NVTX_PAYLOAD_ENUMS (
    -- NVTX payload enum attributes.

    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    schemaId                    INTEGER,                               -- Identifier of the payload schema.
    name                        TEXT,                                  -- Schema name.
    numEntries                  INTEGER,                               -- Number of entries in the enum.
    size                        INTEGER                                -- Size of enumeration type in bytes.
);
CREATE TABLE NVTX_PAYLOAD_ENUM_ENTRIES (
    -- NVTX payload enum entries.

    domainId                    INTEGER   NOT NULL,                    -- User-controlled ID that can be used to group events.
    schemaId                    INTEGER   NOT NULL,                    -- Identifier of the payload schema.
    idx                         INTEGER   NOT NULL,                    -- Index of the entry in the payload schema.
    name                        TEXT,                                  -- Name of the enum value.
    value                       INTEGER,                               -- Value of the enum entry.
    isFlag                      INTEGER                                -- Indicates that the entry sets a specific set of bits, which can be used to define bitsets.
);
CREATE TABLE NVTX_SCOPES (
    -- NVTX scopes.

    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    scopeId                     INTEGER,                               -- Scope ID.
    parentScopeId               INTEGER,                               -- Parent scope ID.
    path                        TEXT                                   -- Scope path.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes transferred (B).
    copyKind                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMCPY_OPER(id)
    deprecatedSrcId             INTEGER,                               -- Deprecated, use srcDeviceId instead.
    srcKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    dstKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    srcDeviceId                 INTEGER,                               -- Source device ID.
    srcContextId                INTEGER,                               -- Source context ID.
    dstDeviceId                 INTEGER,                               -- Destination device ID.
    dstContextId                INTEGER,                               -- Destination context ID.
    migrationCause              INTEGER,                               -- REFERENCES ENUM_CUDA_UNIF_MEM_MIGRATION(id)
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    virtualAddress              INTEGER,                               -- Virtual base address of the page/s being transferred.
    copyCount                   INTEGER                                -- The total number of memcopy operations traced in this record. In CUDA MemcpyBatchAsync APIs, multiple memcpy operations may be batched together for optimization purposes based on certain heuristics.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    value                       INTEGER   NOT NULL,                    -- Value assigned to memory.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes set (B).
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    memKind                     INTEGER                                -- REFERENCES ENUM_CUDA_MEM_KIND(id)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    ChannelID                   INTEGER   NOT NULL,                    -- Channel Id of MemDecompress Operation.
    sourceBytes                 INTEGER   NOT NULL,                    -- Number of source bytes (B).
    NumberOfOperations          INTEGER   NOT NULL                     -- Number of operations.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KERNEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER,                               -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
    qmdBulkReleaseDone          INTEGER,                               -- QMD bulk release done timestamp from CWD events.
    qmdPreexitDone              INTEGER,                               -- QMD pre-exit done timestamp from CWD events.
    qmdLastCtaDone              INTEGER,                               -- QMD last CTA done timestamp from CWD events.
    graphId                     INTEGER,                               -- Kernel graph ID.
    clusterX                    INTEGER,                               -- Cluster X dimension.
    clusterY                    INTEGER,                               -- Cluster Y dimension.
    clusterZ                    INTEGER,                               -- Cluster Z dimension.
    clusterSchedulingPolicy     INTEGER,                               -- Cluster scheduling policy.
    maxPotentialClusterSize     INTEGER,                               -- Maximum potential cluster size.
    maxActiveClusters           INTEGER,                               -- Maximum active clusters.
    sharedMemoryRequestedPercentage   INTEGER,                         -- Shared memory requested percentage.
    tensorSizeMinusOneElements   TEXT      NOT NULL                    -- TMA descriptor tensor size minus one elements array.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- Correlation ID of the synchronization API to which this result is associated.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    deprecatedSyncType          INTEGER,                               -- Deprecated, use syncType instead. For older report, REFERENCES ENUM_CUPTI_SYNC_TYPE(id)
    syncType                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUPTI_SYNC_TYPE(id)
    eventId                     INTEGER   NOT NULL,                    -- Event ID for which the synchronization API is called.
    eventSyncId                 INTEGER                                -- CUDA Event Sync ID to link the synchronization API to associated event record API.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT (
    timestamp                   INTEGER,                               -- The device-side CUDA Event completion timestamp. 0 if not collected.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- Correlation ID of the event record API to which this result is associated.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    eventId                     INTEGER   NOT NULL,                    -- Event ID for which the event record API is called.
    eventSyncId                 INTEGER                                -- CUDA Event Sync ID to link event record API to related synchronization APIs.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE_AND_HOST_LAUNCH (
    -- This table includes both CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE and CUPTI_ACTIVITY_KIND_HOST_LAUNCH events. These are stored as a single internal event type in Nsys reports. For HOST_LAUNCH events, graphNodeId and graphId fields will be NULL.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    graphId                     INTEGER                                -- REFERENCES CUDA_GRAPH_EVENTS(graphId)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    graphId                     INTEGER   NOT NULL,                    -- REFERENCES CUDA_GRAPH_EVENTS(graphId)
    graphExecId                 INTEGER   NOT NULL                     -- REFERENCES CUDA_GRAPH_EVENTS(graphExecId)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_OVERHEAD (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER,                               -- REFERENCES StringIds(id) -- Function name
    overheadType                INTEGER   NOT NULL                     -- REFERENCES ENUM_CUPTI_OVERHEAD_TYPE(id)
);
CREATE TABLE CUDA_HOST_CALLBACK (
    -- Host-side callback functions triggered from CUDA streams. These represent CPU-side execution from CUDA graph host function nodes or cudaLaunchHostFunc() calls. Corresponding device-side activities are stored at the CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE_AND_HOST_LAUNCH table.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER                                -- REFERENCES StringIds(id) -- Function name
);
CREATE TABLE CUDNN_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
);
CREATE TABLE CUBLAS_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
);
CREATE TABLE CUDA_GRAPH_NODE_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    graphNodeId                 INTEGER   NOT NULL,                    -- Graph node ID.
    originalGraphNodeId         INTEGER                                -- Reference to the original graph node ID, if cloned node.
);
CREATE TABLE CUDA_GRAPH_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    graphId                     INTEGER,                               -- Graph ID.
    originalGraphId             INTEGER,                               -- Reference to the original graph ID, if cloned.
    graphExecId                 INTEGER                                -- Executable graph ID.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    callchainId                 INTEGER                                -- REFERENCES CUDA_CALLCHAINS(id)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_BLOCK_TRACE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER,                               -- Device ID.
    correlationId               INTEGER,                               -- Correlation ID of the event record API to which this result is associated.
    nodeId                      INTEGER,                               -- Node ID of the event record API to which this result is associated.
    SMId                        INTEGER,                               -- SM ID of the event on which the particular event was running.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    BlockID                     INTEGER   NOT NULL,                    -- Block ID.
    UGPUId                      INTEGER,                               -- uGPU ID of the event on which the particular event was running.
    CGAId                       INTEGER                                -- CGA ID of the event on which the particular event was running.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_BLOCK_PHASE_TRACE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER,                               -- Device ID.
    correlationId               INTEGER,                               -- Correlation ID of the event record API to which this result is associated.
    nodeId                      INTEGER,                               -- Node ID of the event record API to which this result is associated.
    SMId                        INTEGER,                               -- SM ID of the event on which the particular event was running.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    BlockID                     INTEGER   NOT NULL,                    -- Block ID.
    phase1Timestamp             INTEGER   NOT NULL,                    -- Phase start timestamp.
    phase2Timestamp             INTEGER   NOT NULL,                    -- Phase stop timestamp.
    UGPUId                      INTEGER,                               -- uGPU ID of the event on which the particular event was running.
    CGAId                       INTEGER                                -- CGA ID of the event on which the particular event was running.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_WARP_TRACE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER,                               -- Device ID.
    correlationId               INTEGER,                               -- Correlation ID of the event record API to which this result is associated.
    nodeId                      INTEGER,                               -- Node ID of the event record API to which this result is associated.
    SMId                        INTEGER,                               -- SM ID of the event on which the particular event was running.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    BlockID                     INTEGER   NOT NULL,                    -- Block ID.
    WarpID                      INTEGER   NOT NULL,                    -- Warp ID.
    UGPUId                      INTEGER,                               -- uGPU ID of the event on which the particular event was running.
    CGAId                       INTEGER                                -- CGA ID of the event on which the particular event was running.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_WARP_PHASE_TRACE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER,                               -- Device ID.
    correlationId               INTEGER,                               -- Correlation ID of the event record API to which this result is associated.
    nodeId                      INTEGER,                               -- Node ID of the event record API to which this result is associated.
    SMId                        INTEGER,                               -- SM ID of the event on which the particular event was running.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    BlockID                     INTEGER   NOT NULL,                    -- Block ID.
    WarpID                      INTEGER   NOT NULL,                    -- Warp ID.
    EventName                   INTEGER   NOT NULL,                    -- Event Name.
    InternalEventCount          INTEGER   NOT NULL,                    -- Internal event count.
    eventType                   INTEGER   NOT NULL,                    -- Event type, 0 = range, 1 = marker, 2 = warp start/end
    WarpEventIds                TEXT      NOT NULL,                    -- warp event ids.
    WarpEventTimestampOffsets   TEXT      NOT NULL,                    -- warp event timestamp offsets.
    UGPUId                      INTEGER,                               -- uGPU ID of the event on which the particular event was running.
    CGAId                       INTEGER                                -- CGA ID of the event on which the particular event was running.
);
CREATE TABLE CUDA_UM_CPU_PAGE_FAULT_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    address                     INTEGER   NOT NULL,                    -- Virtual address of the page that faulted.
    originalFaultPc             INTEGER,                               -- Program counter of the CPU instruction that caused the page fault.
    CpuInstruction              INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
    unresolvedFaultPc           INTEGER,                               -- True if the program counter was not resolved.
    sourceFile                  INTEGER,                               -- Source file where the page fault occurred.
    sourceLine                  INTEGER                                -- Source line number that caused the page fault in the source file.
);
CREATE TABLE CUDA_UM_GPU_PAGE_FAULT_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    address                     INTEGER   NOT NULL,                    -- Virtual address of the page that faulted.
    numberOfPageFaults          INTEGER   NOT NULL,                    -- Number of page faults for the same page.
    faultAccessType             INTEGER   NOT NULL                     -- REFERENCES ENUM_CUDA_UNIF_MEM_ACCESS_TYPE(id)
);
CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    address                     INTEGER   NOT NULL,                    -- Virtual address of the allocation/deallocation.
    pc                          INTEGER   NOT NULL,                    -- Program counter of the allocation/deallocation.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes allocated/deallocated (B).
    memKind                     INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    memoryOperationType         INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_DEV_MEM_EVENT_OPER(id)
    name                        TEXT,                                  -- Variable name, if available.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    streamId                    INTEGER,                               -- Stream ID.
    localMemoryPoolAddress      INTEGER,                               -- Base address of the local memory pool used
    localMemoryPoolReleaseThreshold   INTEGER,                         -- Release threshold of the local memory pool used
    localMemoryPoolSize         INTEGER,                               -- Size of the local memory pool used
    localMemoryPoolUtilizedSize   INTEGER,                             -- Utilized size of the local memory pool used
    importedMemoryPoolAddress   INTEGER,                               -- Base address of the imported memory pool used
    importedMemoryPoolProcessId   INTEGER                              -- Process ID of the imported memory pool used
);
CREATE TABLE CUDA_GPU_MEMORY_POOL_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    address                     INTEGER   NOT NULL,                    -- The base virtual address of the memory pool.
    operationType               INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMPOOL_OPER(id)
    poolType                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMPOOL_TYPE(id)
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    minBytesToKeep              INTEGER,                               -- Minimum number of bytes to keep of the memory pool.
    localMemoryPoolReleaseThreshold   INTEGER,                         -- Release threshold of the local memory pool used
    localMemoryPoolSize         INTEGER,                               -- Size of the local memory pool used
    localMemoryPoolUtilizedSize   INTEGER                              -- Utilized size of the local memory pool used
);
CREATE TABLE CUDA_CALLCHAINS (
    id                          INTEGER   NOT NULL,                    -- Part of PRIMARY KEY (id, stackDepth).
    symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
    unresolved                  INTEGER,                               -- True if the symbol was not resolved.
    originalIP                  INTEGER,                               -- Instruction pointer value.
    stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

    PRIMARY KEY (id, stackDepth)
);
CREATE TABLE MPI_RANKS (
    -- Mapping of global thread IDs (gtid) to MPI ranks

    globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    rank                        INTEGER   NOT NULL                     -- MPI rank
);
CREATE TABLE MPI_P2P_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    commHandle                  INTEGER,                               -- MPI communicator handle.
    commUid                     INTEGER,                               -- Globally unique MPI communicator ID.
    tag                         INTEGER,                               -- MPI message tag
    remoteRank                  INTEGER,                               -- MPI remote rank (destination or source)
    size                        INTEGER,                               -- MPI message size in bytes
    requestHandle               INTEGER                                -- MPI request handle.
);
CREATE TABLE MPI_COLLECTIVES_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    commHandle                  INTEGER,                               -- MPI communicator handle.
    commUid                     INTEGER,                               -- Globally unique MPI communicator ID.
    rootRank                    INTEGER,                               -- Root rank in the collective
    size                        INTEGER,                               -- MPI message size in bytes (send size for bidirectional ops)
    recvSize                    INTEGER,                               -- MPI receive size in bytes
    requestHandle               INTEGER                                -- MPI request handle.
);
CREATE TABLE MPI_START_WAIT_EVENTS (
    -- MPI_Start*, MPI_Test* and MPI_Wait*

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    requestHandle               INTEGER                                -- MPI request handle.
);
CREATE TABLE MPI_OTHER_EVENTS (
    -- MPI events without additional parameters

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER                                -- REFERENCES StringIds(id) -- Registered NVTX domain/string
);
CREATE TABLE UCP_WORKERS (
    globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    workerUid                   INTEGER   NOT NULL                     -- UCP worker UID
);
CREATE TABLE UCP_SUBMIT_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    bufferAddr                  INTEGER,                               -- Address of the message buffer
    packedSize                  INTEGER,                               -- Message size (packed) in bytes
    peerWorkerUid               INTEGER,                               -- Peer's UCP worker UID
    tag                         INTEGER                                -- UCP message tag
);
CREATE TABLE UCP_PROGRESS_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    bufferAddr                  INTEGER,                               -- Address of the message buffer
    packedSize                  INTEGER,                               -- Message size (packed) in bytes
    peerWorkerUid               INTEGER,                               -- Peer's UCP worker UID
    tag                         INTEGER                                -- UCP message tag
);
CREATE TABLE UCP_EVENTS (
    -- UCP events without additional parameters

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER                                -- REFERENCES StringIds(id) -- Registered NVTX domain/string
);
CREATE TABLE NVTX_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    eventType                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    rangeId                     INTEGER,                               -- Correlation ID returned from a nvtxRangeStart call.
    category                    INTEGER,                               -- User-controlled ID that can be used to group events.
    color                       INTEGER,                               -- Encoded ARGB color value.
    text                        TEXT,                                  -- Explicit name/text (non-registered string)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    uint64Value                 INTEGER,                               -- One of possible payload value union members.
    int64Value                  INTEGER,                               -- One of possible payload value union members.
    doubleValue                 REAL,                                  -- One of possible payload value union members.
    uint32Value                 INTEGER,                               -- One of possible payload value union members.
    int32Value                  INTEGER,                               -- One of possible payload value union members.
    floatValue                  REAL,                                  -- One of possible payload value union members.
    jsonTextId                  INTEGER,                               -- One of possible payload value union members.
    jsonText                    TEXT,                                  -- One of possible payload value union members.
    binaryData                  TEXT                                   -- Binary payload. See docs for format.
);
CREATE TABLE OPENGL_API (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- First function name
    endNameId                   INTEGER,                               -- REFERENCES StringIds(id) -- Last function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    contextId                   INTEGER,                               -- Context ID.
    gpu                         INTEGER,                               -- GPU index.
    display                     INTEGER                                -- Display ID.
);
CREATE TABLE OPENGL_WORKLOAD (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- First function name
    endNameId                   INTEGER,                               -- REFERENCES StringIds(id) -- Last function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    contextId                   INTEGER,                               -- Context ID.
    gpu                         INTEGER,                               -- GPU index.
    display                     INTEGER                                -- Display ID.
);
CREATE TABLE KHR_DEBUG_EVENTS (
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Debug marker/group text
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    source                      INTEGER,                               -- REFERENCES ENUM_OPENGL_DEBUG_SOURCE(id)
    khrdType                    INTEGER,                               -- REFERENCES ENUM_OPENGL_DEBUG_TYPE(id)
    id                          INTEGER,                               -- KHR event ID.
    severity                    INTEGER,                               -- REFERENCES ENUM_OPENGL_DEBUG_SEVERITY(id)
    correlationId               INTEGER,                               -- ID used to correlate KHR CPU trace to GPU trace.
    context                     INTEGER                                -- Context ID.
);
CREATE TABLE OSRT_API (
    -- OS runtime libraries traced to gather information about low-level userspace APIs.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    nestingLevel                INTEGER,                               -- Zero-base index of the nesting level.
    callchainId                 INTEGER   NOT NULL,                    -- REFERENCES OSRT_CALLCHAINS(id)
    argumentsId                 INTEGER   NOT NULL                     -- REFERENCES OSRT_ARGUMENTS(id) -- Experimental.
);
CREATE TABLE OSRT_CALLCHAINS (
    -- Callchains attached to OSRT events, depending on selected profiling settings.

    id                          INTEGER   NOT NULL,                    -- Part of PRIMARY KEY (id, stackDepth).
    symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
    kernelMode                  INTEGER,                               -- True if kernel mode.
    thumbCode                   INTEGER,                               -- True if thumb code.
    unresolved                  INTEGER,                               -- True if the symbol was not resolved.
    specialEntry                INTEGER,                               -- True if artifical entry added during processing callchain.
    originalIP                  INTEGER,                               -- Instruction pointer value.
    unwindMethod                INTEGER,                               -- REFERENCES ENUM_STACK_UNWIND_METHOD(id)
    stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

    PRIMARY KEY (id, stackDepth)
);
CREATE TABLE OSRT_ARGUMENTS (
    -- Arguments OSRT functions were called with. This is an experimental feature and arguments are collected for some specific functions only. Please avoid relying on the content for now.

    id                          INTEGER   NOT NULL,                    -- Part of PRIMARY KEY (id, argumentIndex).
    value                       INTEGER   NOT NULL,                    -- Value of the argument.
    argumentIndex               INTEGER   NOT NULL,                    -- Zero-base index of the argument.

    PRIMARY KEY (id, argumentIndex)
);
CREATE TABLE OSRT_FILE_ACCESS_EVENTS (
    -- OS Runtime events related to file accesses (opening, closing, reading, and writing).

    fileAccessId                INTEGER   NOT NULL,                    -- REFERENCES OSRT_FILE_ACCESS_DESCRIPTORS(fileAccessId)
    threadId                    INTEGER   NOT NULL,                    -- Thread ID.
    startedAt                   INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    endedAt                     INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventType                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OSRT_FILE_ACCESS_EVENT_TYPE(id)
    apiCallId                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    bytesProcessed              INTEGER   NOT NULL,                    -- Actual bytes read/written.
    context                     TEXT      NOT NULL                     -- Additional information about the event
);
CREATE TABLE OSRT_FILE_ACCESS_DESCRIPTORS (
    -- Metadata of all the file accesses that were made by the OS during the recording.

    fileAccessId                INTEGER   NOT NULL,                    -- File Access Id.
    processId                   INTEGER   NOT NULL,                    -- Process ID.
    openedAt                    INTEGER   NOT NULL,                    -- The time when the file was opened (ns).
    closedAt                    INTEGER   NOT NULL,                    -- The time when the file was closed (ns).
    filePath                    TEXT      NOT NULL                     -- The opened file path.
);
CREATE TABLE PROFILER_OVERHEAD (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    overheadType                INTEGER   NOT NULL                     -- REFERENCES ENUM_CUPTI_OVERHEAD_TYPE(id)
);
CREATE TABLE SCHED_EVENTS (
    -- Thread scheduling events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    cpu                         INTEGER   NOT NULL,                    -- ID of CPU this thread was scheduled in or out.
    isSchedIn                   INTEGER   NOT NULL,                    -- 0 if thread was scheduled out, non-zero otherwise.
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    threadState                 INTEGER,                               -- REFERENCES ENUM_SAMPLING_THREAD_STATE(id)
    threadBlock                 INTEGER                                -- REFERENCES ENUM_SCHEDULING_THREAD_BLOCK(id)
);
CREATE TABLE COMPOSITE_EVENTS (
    -- Thread sampling events.

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the composite event.
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    cpu                         INTEGER,                               -- ID of CPU this thread was running on.
    threadState                 INTEGER,                               -- REFERENCES ENUM_SAMPLING_THREAD_STATE(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    cpuCycles                   INTEGER   NOT NULL                     -- Value of Performance Monitoring Unit (PMU) counter.
);
CREATE TABLE SAMPLING_CALLCHAINS (
    -- Callchain entries obtained from composite events, used to construct function table views.

    id                          INTEGER   NOT NULL,                    -- REFERENCES COMPOSITE_EVENTS(id)
    symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
    kernelMode                  INTEGER,                               -- True if kernel mode.
    thumbCode                   INTEGER,                               -- True if thumb code.
    unresolved                  INTEGER,                               -- True if the symbol was not resolved.
    specialEntry                INTEGER,                               -- True if artifical entry added during processing callchain.
    originalIP                  INTEGER,                               -- Instruction pointer value.
    unwindMethod                INTEGER,                               -- REFERENCES ENUM_STACK_UNWIND_METHOD(id)
    stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

    PRIMARY KEY (id, stackDepth)
);
CREATE TABLE PERF_EVENT_SOC_OR_CPU_RAW_EVENT (
    -- SoC and CPU raw event values from Sampled Performance Counters.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    vmId                        INTEGER,                               -- VM ID.
    componentId                 INTEGER,                               -- REFERENCES TARGET_INFO_COMPONENT(componentId)
    eventId                     INTEGER,                               -- REFERENCES TARGET_INFO_PERF_METRIC(id)
    count                       INTEGER                                -- Counter data value
);
CREATE TABLE PERF_EVENT_SOC_OR_CPU_METRIC_EVENT (
    -- SoC and CPU metric values from Sampled Performance Counters.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    vmId                        INTEGER,                               -- VM ID.
    componentId                 INTEGER,                               -- REFERENCES TARGET_INFO_COMPONENT(componentId)
    metricId                    INTEGER,                               -- REFERENCES TARGET_INFO_PERF_METRIC(id)
    value                       REAL                                   -- Metric data value
);
CREATE TABLE DX12_API (
    id                          INTEGER   NOT NULL   PRIMARY KEY,
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    shortContextId              INTEGER,                               -- Short form of the COM interface object address.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- PIX marker text
    commandListType             INTEGER,                               -- REFERENCES ENUM_D3D12_CMD_LIST_TYPE(id)
    objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- D3D12 object name
    longContextId               INTEGER                                -- Long form of the COM interface object address.
);
CREATE TABLE DX12_WORKLOAD (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    shortContextId              INTEGER,                               -- Short form of the COM interface object address.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    gpu                         INTEGER,                               -- GPU index.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- PIX marker text
    commandListType             INTEGER,                               -- REFERENCES ENUM_D3D12_CMD_LIST_TYPE(id)
    objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- D3D12 object name
    longContextId               INTEGER                                -- Long form of the COM interface object address.
);
CREATE TABLE DX12_MEMORY_OPERATION (
    gpu                         INTEGER,                               -- GPU index.
    rangeStart                  INTEGER,                               -- Time offset denoting the beginning of a memory range (B).
    rangeEnd                    INTEGER,                               -- Time offset denoting the end of a memory range (B).
    subresourceId               INTEGER,                               -- Subresource index.
    heapType                    INTEGER,                               -- REFERENCES ENUM_D3D12_HEAP_TYPE(id)
    heapFlags                   INTEGER,                               -- REFERENCES ENUM_D3D12_HEAP_FLAGS(id)
    cpuPageProperty             INTEGER,                               -- REFERENCES ENUM_D3D12_PAGE_PROPERTY(id)
    nvApiFlags                  INTEGER,                               -- NV specific flags. See docs for specifics.
    objectHandle                INTEGER,                               -- Handle to the graphics object created or modified by the memory operation.
    bindTargetHandle            INTEGER,                               -- Handle to the target resource for bind operations.
    memorySize                  INTEGER,                               -- Size of the memory allocation (B).
    memoryOffset                INTEGER,                               -- Offset within the memory allocation (B).
    resourceFlags               INTEGER,                               -- Combination of D3D12_RESOURCE_FLAGS enum values specifying resource usage options.
    dimension                   INTEGER,                               -- D3D12_RESOURCE_DIMENSION enum value specifying the resource type.
    traceEventId                INTEGER   NOT NULL                     -- REFERENCES DX12_API(id)
);
CREATE TABLE DXGI_API (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    shortContextId              INTEGER,                               -- Short form of the COM interface object address.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER                                -- REFERENCES StringIds(id) -- PIX marker text
);
CREATE TABLE NVAPI_API (
    id                          INTEGER   NOT NULL   PRIMARY KEY,
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    longContextId               INTEGER                                -- Long form of the COM interface object address.
);
CREATE TABLE NVAPI_MEMORY_OPERATION (
    rangeStart                  INTEGER,                               -- Time offset denoting the beginning of a memory range (B).
    rangeEnd                    INTEGER,                               -- Time offset denoting the end of a memory range (B).
    subresourceId               INTEGER,                               -- Subresource index.
    heapType                    INTEGER,                               -- REFERENCES ENUM_D3D12_HEAP_TYPE(id)
    heapFlags                   INTEGER,                               -- REFERENCES ENUM_D3D12_HEAP_FLAGS(id)
    memoryProperty              INTEGER,                               -- REFERENCES ENUM_D3D12_PAGE_PROPERTY(id)
    nvApiFlags                  INTEGER,                               -- NV specific flags. See docs for specifics.
    traceEventId                INTEGER   NOT NULL                     -- REFERENCES NVAPI_API(id)
);
CREATE TABLE VULKAN_API (
    id                          INTEGER   NOT NULL   PRIMARY KEY,
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    contextId                   INTEGER,                               -- Short form of the interface object address.
    objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- Vulkan object name
    interfaceAddress            INTEGER                                -- Interface object address.
);
CREATE TABLE VULKAN_WORKLOAD (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    gpu                         INTEGER,                               -- GPU index.
    contextId                   INTEGER,                               -- Short form of the interface object address.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER                                -- REFERENCES StringIds(id) -- Vulkan CPU debug marker string
);
CREATE TABLE VULKAN_DEBUG_API (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    contextId                   INTEGER,                               -- Short form of the interface object address.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER                                -- REFERENCES StringIds(id) -- Vulkan CPU debug marker string
);
CREATE TABLE VULKAN_PIPELINE_CREATION_EVENTS (
    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the pipeline creation event.
    duration                    INTEGER,                               -- Event duration (ns).
    flags                       INTEGER,                               -- REFERENCES ENUM_VULKAN_PIPELINE_CREATION_FLAGS(id)
    traceEventId                INTEGER   NOT NULL                     -- REFERENCES VULKAN_API(id) -- ID of the attached vulkan API.
);
CREATE TABLE VULKAN_PIPELINE_STAGE_EVENTS (
    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the pipeline stage event.
    duration                    INTEGER,                               -- Event duration (ns).
    flags                       INTEGER,                               -- REFERENCES ENUM_VULKAN_PIPELINE_CREATION_FLAGS(id)
    creationEventId             INTEGER   NOT NULL                     -- REFERENCES VULKAN_PIPELINE_CREATION_EVENTS(id) -- ID of the attached pipeline creation event.
);
CREATE TABLE VULKAN_MEMORY_OPERATION (
    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the memory operation.
    gpu                         INTEGER,                               -- GPU index.
    rangeStart                  INTEGER,                               -- Time offset denoting the beginning of a memory range (B).
    rangeEnd                    INTEGER,                               -- Time offset denoting the end of a memory range (B).
    contextId                   INTEGER,                               -- Interface object address.
    objectHandle                INTEGER,                               -- Handle to the graphics object created or modified by the memory operation.
    memorySize                  INTEGER,                               -- Size of the memory allocation (B).
    memoryOffset                INTEGER,                               -- Offset within the memory allocation (B).
    bindTargetHandle            INTEGER,                               -- Handle to the target resource for bind operations.
    traceEventId                INTEGER   NOT NULL                     -- REFERENCES VULKAN_API(id)
);
CREATE TABLE VULKAN_MEMORY_TYPES (
    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the memory type entry.
    heapIndex                   INTEGER,                               -- Index of the heap that this memory type references.
    heapFlags                   INTEGER,                               -- REFERENCES ENUM_VULKAN_HEAP_FLAGS(id)
    memoryProperty              INTEGER,                               -- REFERENCES ENUM_VULKAN_MEMORY_PROPERTY_FLAGS(id)
    heapType                    INTEGER,                               -- REFERENCES ENUM_VULKAN_HEAP_TYPE(id)
    memOpId                     INTEGER   NOT NULL                     -- REFERENCES VULKAN_MEMORY_OPERATION(id) -- ID of the parent memory operation.
);
CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS (
    tag                         INTEGER   NOT NULL,                    -- REFERENCES ENUM_GPU_CTX_SWITCH(id)
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    seqNo                       INTEGER   NOT NULL,                    -- Sequential event number.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    timestamp                   INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    gpuId                       INTEGER                                -- GPU index.
);
CREATE TABLE OPENMP_EVENT_KIND_THREAD (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    threadId                    INTEGER,                               -- Internal thread sequence starting from 1.
    threadType                  INTEGER                                -- REFERENCES ENUM_OPENMP_THREAD(id)
);
CREATE TABLE OPENMP_EVENT_KIND_PARALLEL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- Internal parallel region sequence starting from 1.
    parentTaskId                INTEGER                                -- ID for task that creates this parallel region.
);
CREATE TABLE OPENMP_EVENT_KIND_SYNC_REGION_WAIT (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER,                               -- ID of the task that this event belongs to.
    kind                        INTEGER                                -- REFERENCES ENUM_OPENMP_SYNC_REGION(id)
);
CREATE TABLE OPENMP_EVENT_KIND_SYNC_REGION (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER,                               -- ID of the task that this event belongs to.
    kind                        INTEGER                                -- REFERENCES ENUM_OPENMP_SYNC_REGION(id)
);
CREATE TABLE OPENMP_EVENT_KIND_TASK (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER,                               -- ID of the task that this event belongs to.
    kind                        INTEGER                                -- REFERENCES ENUM_OPENMP_TASK_FLAG(id)
);
CREATE TABLE OPENMP_EVENT_KIND_MASTER (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER                                -- ID of the task that this event belongs to.
);
CREATE TABLE OPENMP_EVENT_KIND_REDUCTION (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER                                -- ID of the task that this event belongs to.
);
CREATE TABLE OPENMP_EVENT_KIND_TASK_CREATE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parentTaskId                INTEGER,                               -- ID of the parent task that is creating a new task.
    newTaskId                   INTEGER                                -- ID of the new task that is being created.
);
CREATE TABLE OPENMP_EVENT_KIND_TASK_SCHEDULE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    priorTaskId                 INTEGER,                               -- ID of the task that is being switched out.
    priorTaskStatus             INTEGER,                               -- REFERENCES ENUM_OPENMP_TASK_STATUS(id)
    nextTaskId                  INTEGER                                -- ID of the task that is being switched in.
);
CREATE TABLE OPENMP_EVENT_KIND_CANCEL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    taskId                      INTEGER                                -- ID of the task that is being cancelled.
);
CREATE TABLE OPENMP_EVENT_KIND_MUTEX_WAIT (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
    waitId                      INTEGER,                               -- ID indicating the object being waited.
    taskId                      INTEGER                                -- ID of the task that this event belongs to.
);
CREATE TABLE OPENMP_EVENT_KIND_CRITICAL_SECTION (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
    waitId                      INTEGER                                -- ID indicating the object being held.
);
CREATE TABLE OPENMP_EVENT_KIND_MUTEX_RELEASED (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
    waitId                      INTEGER,                               -- ID indicating the object being released.
    taskId                      INTEGER                                -- ID of the task that this event belongs to.
);
CREATE TABLE OPENMP_EVENT_KIND_LOCK_INIT (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
    waitId                      INTEGER                                -- ID indicating object being created/destroyed.
);
CREATE TABLE OPENMP_EVENT_KIND_LOCK_DESTROY (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
    waitId                      INTEGER                                -- ID indicating object being created/destroyed.
);
CREATE TABLE OPENMP_EVENT_KIND_WORKSHARE (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_WORK(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER,                               -- ID of the task that this event belongs to.
    count                       INTEGER                                -- Measure of the quantity of work involved in the region.
);
CREATE TABLE OPENMP_EVENT_KIND_DISPATCH (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_DISPATCH(id)
    parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
    taskId                      INTEGER                                -- ID of the task that this event belongs to.
);
CREATE TABLE OPENMP_EVENT_KIND_FLUSH (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- Currently unused.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
    threadId                    INTEGER                                -- ID of the thread that this event belongs to.
);
CREATE TABLE D3D11_PIX_DEBUG_API (
    -- D3D11 debug marker events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    shortContextId              INTEGER,                               -- Short form of the COM interface object address.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER                                -- REFERENCES StringIds(id) -- PIX marker text
);
CREATE TABLE D3D12_PIX_DEBUG_API (
    -- D3D12 debug marker events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
    endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    shortContextId              INTEGER,                               -- Short form of the COM interface object address.
    frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
    color                       INTEGER,                               -- Encoded ARGB color value.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- PIX marker text
    commandListType             INTEGER,                               -- REFERENCES ENUM_D3D12_CMD_LIST_TYPE(id)
    objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- D3D12 object name
    longContextId               INTEGER                                -- Long form of the COM interface object address.
);
CREATE TABLE WDDM_EVICT_ALLOCATION_EVENTS (
    -- Raw ETW EvictAllocation events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    allocationHandle            INTEGER   NOT NULL                     -- Global allocation handle.
);
CREATE TABLE WDDM_PAGING_QUEUE_PACKET_START_EVENTS (
    -- Raw ETW PagingQueuePacketStart events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    dxgDevice                   INTEGER,                               -- Address of an IDXGIDevice.
    dxgAdapter                  INTEGER,                               -- Address of an IDXGIAdapter.
    pagingQueue                 INTEGER   NOT NULL,                    -- Address of the paging queue.
    pagingQueuePacket           INTEGER   NOT NULL,                    -- Address of the paging queue packet.
    sequenceId                  INTEGER   NOT NULL,                    -- Internal sequence starting from 0.
    alloc                       INTEGER,                               -- Allocation handle.
    vidMmOpType                 INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_VIDMM_OP_TYPE(id)
    pagingQueueType             INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_PAGING_QUEUE_TYPE(id)
);
CREATE TABLE WDDM_PAGING_QUEUE_PACKET_STOP_EVENTS (
    -- Raw ETW PagingQueuePacketStop events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    pagingQueue                 INTEGER   NOT NULL,                    -- Address of the paging queue.
    pagingQueuePacket           INTEGER   NOT NULL,                    -- Address of the paging queue packet.
    sequenceId                  INTEGER   NOT NULL                     -- Internal sequence starting from 0.
);
CREATE TABLE WDDM_PAGING_QUEUE_PACKET_INFO_EVENTS (
    -- Raw ETW PagingQueuePacketInfo events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    pagingQueue                 INTEGER   NOT NULL,                    -- Address of the paging queue.
    pagingQueuePacket           INTEGER   NOT NULL,                    -- Address of the paging queue packet.
    sequenceId                  INTEGER   NOT NULL                     -- Internal sequence starting from 0.
);
CREATE TABLE WDDM_QUEUE_PACKET_START_EVENTS (
    -- Raw ETW QueuePacketStart events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    dmaBufferSize               INTEGER   NOT NULL,                    -- The dma buffer size.
    dmaBuffer                   INTEGER   NOT NULL,                    -- The reported address of dma buffer.
    queuePacket                 INTEGER   NOT NULL,                    -- The address of queue packet.
    progressFenceValue          INTEGER   NOT NULL,                    -- The fence value.
    packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
    submitSequence              INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
    allocationListSize          INTEGER   NOT NULL,                    -- The number of allocations referenced.
    patchLocationListSize       INTEGER   NOT NULL,                    -- The number of patch locations.
    present                     INTEGER   NOT NULL,                    -- True or False if the packet is a present packet.
    engineType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
    syncObject                  INTEGER                                -- The address of fence object.
);
CREATE TABLE WDDM_QUEUE_PACKET_STOP_EVENTS (
    -- Raw ETW QueuePacketStop events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    queuePacket                 INTEGER   NOT NULL,                    -- The address of queue packet.
    packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
    submitSequence              INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
    preempted                   INTEGER   NOT NULL,                    -- True or False if the packet is preempted.
    timeouted                   INTEGER   NOT NULL,                    -- True or False if the packet is timeouted.
    engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
);
CREATE TABLE WDDM_QUEUE_PACKET_INFO_EVENTS (
    -- Raw ETW QueuePacketInfo events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
    submitSequence              INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
    engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
);
CREATE TABLE WDDM_DMA_PACKET_START_EVENTS (
    -- Raw ETW DmaPacketStart events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    queuePacketContext          INTEGER   NOT NULL,                    -- The queue packet context.
    uliSubmissionId             INTEGER   NOT NULL,                    -- The queue packet submission ID.
    dmaBuffer                   INTEGER   NOT NULL,                    -- The reported address of dma buffer.
    packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
    ulQueueSubmitSequence       INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
    quantumStatus               INTEGER   NOT NULL,                    -- The quantum Status.
    engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
);
CREATE TABLE WDDM_DMA_PACKET_STOP_EVENTS (
    -- Raw ETW DmaPacketStop events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    uliCompletionId             INTEGER   NOT NULL,                    -- The queue packet completion ID.
    packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
    ulQueueSubmitSequence       INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
    preempted                   INTEGER   NOT NULL,                    -- True or False if the packet is preempted.
    engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
);
CREATE TABLE WDDM_DMA_PACKET_INFO_EVENTS (
    -- Raw ETW DmaPacketInfo events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    uliCompletionId             INTEGER   NOT NULL,                    -- The queue packet completion ID.
    faultedVirtualAddress       INTEGER   NOT NULL,                    -- The virtual address of faulted process.
    faultedProcessHandle        INTEGER   NOT NULL,                    -- The address of faulted process.
    packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
    ulQueueSubmitSequence       INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
    interruptType               INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_INTERRUPT_TYPE(id)
    quantumStatus               INTEGER   NOT NULL,                    -- The quantum Status.
    pageFaultFlags              INTEGER   NOT NULL,                    -- The page fault flag ID.
    engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
);
CREATE TABLE WDDM_HW_QUEUE_EVENTS (
    -- Raw ETW HwQueueStart events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
    hwQueue                     INTEGER   NOT NULL,                    -- The address of HW queue.
    parentDxgHwQueue            INTEGER   NOT NULL                     -- The address of parent Dxg HW queue.
);
CREATE TABLE NVVIDEO_ENCODER_API (
    -- NV Video Encoder API traced to gather information about NVIDIA Video Codek SDK Encoder APIs.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    apiId                       INTEGER                                -- REFERENCES GPU_VIDEO_ENGINE_WORKLOAD(apiId)
);
CREATE TABLE NVVIDEO_DECODER_API (
    -- NV Video Encoder API traced to gather information about NVIDIA Video Codek SDK Decoder APIs.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    apiId                       INTEGER                                -- REFERENCES GPU_VIDEO_ENGINE_WORKLOAD(apiId)
);
CREATE TABLE NVVIDEO_JPEG_API (
    -- NV Video Encoder API traced to gather information about NVIDIA Video Codek SDK JPEG APIs.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
);
CREATE TABLE GPU_VIDEO_ENGINE_WORKLOAD (
    -- Video engine workload events

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalEngineId              INTEGER   NOT NULL,                    -- Serialized GlobalId.
    engineType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_VIDEO_ENGINE_TYPE(id)
    engineId                    INTEGER   NOT NULL,
    vmId                        INTEGER   NOT NULL,                    -- Driver provided ID.
    contextId                   INTEGER,                               -- Context ID.
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    apiId                       INTEGER   NOT NULL,                    -- ID used to correlate API and workload trace.
    codecId                     INTEGER                                -- REFERENCES ENUM_VIDEO_ENGINE_CODEC(id)
);
CREATE TABLE GPU_VIDEO_ENGINE_MISSING (
    -- Video engine missing ranges

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalEngineId              INTEGER   NOT NULL,                    -- Serialized GlobalId.
    rangeCount                  INTEGER   NOT NULL                     -- Number of missing ranges.
);
CREATE TABLE MEMORY_TRANSFER_EVENTS (
    -- Raw ETW Memory Transfer events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    gpu                         INTEGER,                               -- GPU index.
    taskId                      INTEGER   NOT NULL,                    -- The event task ID.
    eventId                     INTEGER   NOT NULL,                    -- Event ID.
    allocationGlobalHandle      INTEGER   NOT NULL,                    -- Address of the global allocation handle.
    dmaBuffer                   INTEGER   NOT NULL,                    -- The reported address of dma buffer.
    size                        INTEGER   NOT NULL,                    -- The size of the dma buffer in bytes.
    offset                      INTEGER   NOT NULL,                    -- The offset from the start of the reported dma buffer in bytes.
    memoryTransferType          INTEGER   NOT NULL                     -- REFERENCES ENUM_ETW_MEMORY_TRANSFER_TYPE(id)
);
CREATE TABLE NV_LOAD_BALANCE_MASTER_EVENTS (
    -- Raw ETW NV-wgf2um LoadBalanceMaster events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    eventId                     INTEGER   NOT NULL,                    -- Event ID.
    task                        TEXT      NOT NULL,                    -- The task name.
    frameCount                  INTEGER   NOT NULL,                    -- The frame ID.
    frameTime                   REAL      NOT NULL,                    -- Frame duration.
    averageFrameTime            REAL      NOT NULL,                    -- Average of frame duration.
    averageLatency              REAL      NOT NULL,                    -- Average of latency.
    minLatency                  REAL      NOT NULL,                    -- The minimum latency.
    averageQueuedFrames         REAL      NOT NULL,                    -- Average number of queued frames.
    totalActiveMs               REAL      NOT NULL,                    -- Total active time in milliseconds.
    totalIdleMs                 REAL      NOT NULL,                    -- Total idle time in milliseconds.
    idlePercent                 REAL      NOT NULL,                    -- The percentage of idle time.
    isGPUAlmostOneFrameAhead    INTEGER   NOT NULL                     -- True or False if GPU is almost one frame ahead.
);
CREATE TABLE NV_LOAD_BALANCE_EVENTS (
    -- Raw ETW NV-wgf2um LoadBalance events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
    gpu                         INTEGER   NOT NULL,                    -- GPU index.
    eventId                     INTEGER   NOT NULL,                    -- Event ID.
    task                        TEXT      NOT NULL,                    -- The task name.
    averageFPS                  REAL      NOT NULL,                    -- Average frame per second.
    queuedFrames                REAL      NOT NULL,                    -- The amount of queued frames.
    averageQueuedFrames         REAL      NOT NULL,                    -- Average number of queued frames.
    currentCPUTime              REAL      NOT NULL,                    -- The current CPU time.
    averageCPUTime              REAL      NOT NULL,                    -- Average CPU time.
    averageStallTime            REAL      NOT NULL,                    -- Average of stall time.
    averageCPUIdleTime          REAL      NOT NULL,                    -- Average CPU idle time.
    isGPUAlmostOneFrameAhead    INTEGER   NOT NULL                     -- True or False if GPU is almost one frame ahead.
);
CREATE TABLE PROCESSES (
    -- Names and identifiers of processes captured in the report.

    globalPid                   INTEGER,                               -- Serialized GlobalId.
    pid                         INTEGER,                               -- The process ID.
    name                        TEXT                                   -- The process name.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_DATA (
    -- OpenACC data events collected using CUPTI.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    eventKind                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_EVENT_KIND(id)
    DeviceType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_DEVICE(id)
    lineNo                      INTEGER   NOT NULL,                    -- Line number of the directive or program construct.
    cuDeviceId                  INTEGER   NOT NULL,                    -- CUDA device ID. Valid only if deviceType is acc_device_nvidia.
    cuContextId                 INTEGER   NOT NULL,                    -- CUDA context ID. Valid only if deviceType is acc_device_nvidia.
    cuStreamId                  INTEGER   NOT NULL,                    -- CUDA stream ID. Valid only if deviceType is acc_device_nvidia.
    srcFile                     INTEGER,                               -- REFERENCES StringIds(id) -- Source file name or path
    funcName                    INTEGER,                               -- REFERENCES StringIds(id) -- Function in which event occurred
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    bytes                       INTEGER,                               -- Number of bytes.
    varName                     INTEGER                                -- REFERENCES StringIds(id) -- Variable name
);
CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH (
    -- OpenACC launch events collected using CUPTI.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    eventKind                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_EVENT_KIND(id)
    DeviceType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_DEVICE(id)
    lineNo                      INTEGER   NOT NULL,                    -- Line number of the directive or program construct.
    cuDeviceId                  INTEGER   NOT NULL,                    -- CUDA device ID. Valid only if deviceType is acc_device_nvidia.
    cuContextId                 INTEGER   NOT NULL,                    -- CUDA context ID. Valid only if deviceType is acc_device_nvidia.
    cuStreamId                  INTEGER   NOT NULL,                    -- CUDA stream ID. Valid only if deviceType is acc_device_nvidia.
    srcFile                     INTEGER,                               -- REFERENCES StringIds(id) -- Source file name or path
    funcName                    INTEGER,                               -- REFERENCES StringIds(id) -- Function in which event occurred
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    numGangs                    INTEGER,                               -- Number of gangs created for this kernel launch.
    numWorkers                  INTEGER,                               -- Number of workers created for this kernel launch.
    vectorLength                INTEGER,                               -- Number of vector lanes created for this kernel launch.
    kernelName                  INTEGER                                -- REFERENCES StringIds(id) -- Kernel name
);
CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_OTHER (
    -- OpenACC other events collected using CUPTI.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    eventKind                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_EVENT_KIND(id)
    DeviceType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_DEVICE(id)
    lineNo                      INTEGER   NOT NULL,                    -- Line number of the directive or program construct.
    cuDeviceId                  INTEGER   NOT NULL,                    -- CUDA device ID. Valid only if deviceType is acc_device_nvidia.
    cuContextId                 INTEGER   NOT NULL,                    -- CUDA context ID. Valid only if deviceType is acc_device_nvidia.
    cuStreamId                  INTEGER   NOT NULL,                    -- CUDA stream ID. Valid only if deviceType is acc_device_nvidia.
    srcFile                     INTEGER,                               -- REFERENCES StringIds(id) -- Source file name or path
    funcName                    INTEGER,                               -- REFERENCES StringIds(id) -- Function in which event occurred
    correlationId               INTEGER                                -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
);
CREATE TABLE NET_NIC_METRIC (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalId                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
    portId                      INTEGER   NOT NULL,                    -- REFERENCES NET_IB_DEVICE_PORT_INFO(portNumber) -- Port ID
    metricsListId               INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsListId)
    metricsIdx                  INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsIdx)
    value                       INTEGER   NOT NULL                     -- Counter data value
);
CREATE TABLE NET_IB_SWITCH_METRIC (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalId                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
    portId                      INTEGER   NOT NULL,                    -- REFERENCES NET_IB_DEVICE_PORT_INFO(portNumber) -- Port ID
    metricsListId               INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsListId)
    metricsIdx                  INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsIdx)
    value                       INTEGER   NOT NULL                     -- Counter data value
);
CREATE TABLE NET_IB_SWITCH_CONGESTION_EVENT (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalId                    INTEGER   NOT NULL,                    -- Serialized GlobalId (view in hex).
    congestionType              INTEGER,                               -- REFERENCES ENUM_NET_IB_CONGESTION_EVENT_TYPE(id)
    packetSLID                  INTEGER,                               -- Packet Source LID
    packetDLID                  INTEGER,                               -- Packet Destination LID
    packetSL                    INTEGER,                               -- Packet Service Level
    packetOpCode                INTEGER,                               -- Packet Operation Code
    packetSourceQP              INTEGER,                               -- Packet Source Queue Pair
    packetDestinationQP         INTEGER,                               -- Packet Destination Queue Pair
    switchIngressPort           INTEGER,                               -- Packet's Ingress Switch Port
    switchEgressPort            INTEGER                                -- Packet's Egress Switch Port
);
CREATE TABLE PMU_EVENTS (
    -- CPU Core events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalVm                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
    cpu                         INTEGER   NOT NULL,                    -- CPU ID
    counter_id                  INTEGER                                -- REFERENCES PMU_EVENT_COUNTERS(id)
);
CREATE TABLE PMU_EVENT_COUNTERS (
    -- CPU Core events counters.

    id                          INTEGER   NOT NULL,
    idx                         INTEGER   NOT NULL,                    -- REFERENCES PMU_EVENT_REQUESTS(id).
    value                       INTEGER   NOT NULL                     -- Counter data value
);
CREATE TABLE TRACE_PROCESS_EVENT_NVMEDIA (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    correlationId               INTEGER                                -- First ID matching an API call to GPU workloads.
);
CREATE TABLE TEGRA_INTERNAL_API_CALLS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
);
CREATE TABLE UNCORE_PMU_EVENTS (
    -- PMU Uncore events.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalVm                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
    clusterId                   INTEGER,                               -- Cluster ID.
    counterId                   INTEGER                                -- REFERENCES UNCORE_PMU_EVENT_VALUES(id).
);
CREATE TABLE UNCORE_PMU_EVENT_VALUES (
    -- Uncore events values.

    id                          INTEGER   NOT NULL,
    type                        INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    value                       INTEGER   NOT NULL,                    -- Event value.
    rawId                       INTEGER   NOT NULL,                    -- Event value raw ID.
    clusterId                   INTEGER                                -- Cluster ID.
);
CREATE TABLE DIAGNOSTIC_EVENT (
    timestamp                   INTEGER   NOT NULL,                    -- Event timestamp (ns).
    timestampType               INTEGER   NOT NULL,                    -- REFERENCES ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE(id)
    source                      INTEGER   NOT NULL,                    -- REFERENCES ENUM_DIAGNOSTIC_SOURCE_TYPE(id)
    severity                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id)
    text                        TEXT      NOT NULL,                    -- Diagnostic message text
    globalPid                   INTEGER                                -- Serialized GlobalId.
);
CREATE TABLE SYSCALL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    callchainId                 INTEGER   NOT NULL                     -- REFERENCES SYSCALL_CALLCHAINS(id)
);
CREATE TABLE SYSCALL_CALLCHAINS (
    -- Callchains attached to syscall events, depending on selected profiling settings.

    id                          INTEGER   NOT NULL,                    -- Part of PRIMARY KEY (id, stackDepth).
    stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.
    symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
    unresolved                  INTEGER,                               -- True if the symbol was not resolved.

    PRIMARY KEY (id, stackDepth)
);
CREATE TABLE BANDWIDTH_USAGE_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    globalVm                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
    valuesId                    INTEGER                                -- REFERENCES BANDWIDTH_USAGE_VALUES(id)
);
CREATE TABLE BANDWIDTH_USAGE_VALUES (
    -- BandwidthUsage event XmcClient values.

    id                          INTEGER   NOT NULL,
    idx                         INTEGER   NOT NULL,
    value                       INTEGER   NOT NULL,                    -- Counter data value

    PRIMARY KEY (id, idx)
);
```

Note

GENERIC_EVENTS.typeId is a composite bit field that combines HW ID, VM ID, source ID, and type ID with the following structure:

\<Hardware ID:8\>\<VM ID:8\>\<Source ID:16\>\<Type ID:32\>

The type ID is yet another composite bit field that combines the GPU metrics event tag and the GPU ID. To extract the latter, you need to get the lower 8 bits:

SELECT typeId & 0xFF AS gpuId FROM GENERIC_EVENTS

Some event types have been deprecated and are no longer supported by Nsight Systems. While tables for these event will no longer appear in exported SQL databases, databases exported by older versions of Nsight Systems may still contain them.

``` sql
CREATE TABLE ETW_EVENTS_DEPRECATED_TABLE (
    [...]
);
CREATE TABLE GPU_MEMORY_BUDGET_EVENTS (
    -- Raw ETW VidMmProcessBudgetChange events (deprecated).

    [...]
);
CREATE TABLE GPU_MEMORY_USAGE_EVENTS (
    -- Raw ETW VidMmProcessUsageChange events (deprecated).

    [...]
);
CREATE TABLE DEMOTED_BYTES_EVENTS (
    -- Raw ETW VidMmProcessDemotedCommitmentChange events (deprecated).

    [...]
);
CREATE TABLE TOTAL_BYTES_RESIDENT_IN_SEGMENT_EVENTS (
    -- Raw ETW TotalBytesResidentInSegment events (deprecated).

    [...]
);
```

### SQLite Schema Event Values

Here are the set values stored in enums in the Nsight Systems SQLite schema

**CUDA Memcopy Kind**

``` cpp
0 - CUDA_MEMCPY_KIND_UNKNOWN
1 - CUDA_MEMCPY_KIND_HTOD
2 - CUDA_MEMCPY_KIND_DTOH
3 - CUDA_MEMCPY_KIND_HTOA
4 - CUDA_MEMCPY_KIND_ATOH
5 - CUDA_MEMCPY_KIND_ATOA
6 - CUDA_MEMCPY_KIND_ATOD
7 - CUDA_MEMCPY_KIND_DTOA
8 - CUDA_MEMCPY_KIND_DTOD
9 - CUDA_MEMCPY_KIND_HTOH
10 - CUDA_MEMCPY_KIND_PTOP
11 - CUDA_MEMCPY_KIND_UVM_HTOD
12 - CUDA_MEMCPY_KIND_UVM_DTOH
13 - CUDA_MEMCPY_KIND_UVM_DTOD
```

**CUDA Memory Operations Memory Kind**

``` cpp
0 - CUDA_MEMOPR_MEMORY_KIND_PAGEABLE
1 - CUDA_MEMOPR_MEMORY_KIND_PINNED
2 - CUDA_MEMOPR_MEMORY_KIND_DEVICE
3 - CUDA_MEMOPR_MEMORY_KIND_ARRAY
4 - CUDA_MEMOPR_MEMORY_KIND_MANAGED
5 - CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC
6 - CUDA_MEMOPR_MEMORY_KIND_MANAGED_STATIC
7 - CUDA_MEMOPR_MEMORY_KIND_UNKNOWN
```

**CUDA Event Class Values**

``` cpp
0 - TRACE_PROCESS_EVENT_CUDA_RUNTIME
1 - TRACE_PROCESS_EVENT_CUDA_DRIVER
13 - TRACE_PROCESS_EVENT_CUDA_EGL_DRIVER
28 - TRACE_PROCESS_EVENT_CUDNN
29 - TRACE_PROCESS_EVENT_CUBLAS
33 - TRACE_PROCESS_EVENT_CUDNN_START
34 - TRACE_PROCESS_EVENT_CUDNN_FINISH
35 - TRACE_PROCESS_EVENT_CUBLAS_START
36 - TRACE_PROCESS_EVENT_CUBLAS_FINISH
67 - TRACE_PROCESS_EVENT_CUDABACKTRACE
77 - TRACE_PROCESS_EVENT_CUDA_GRAPH_NODE_CREATION
```

See [CUPTI documentation](https://docs.nvidia.com/cupti/index.html) for detailed information on collected event and data types.

**NVTX Event Type Values**

``` cpp
33 - NvtxCategory
34 - NvtxMark
39 - NvtxThread
59 - NvtxPushPopRange
60 - NvtxStartEndRange
75 - NvtxDomainCreate
76 - NvtxDomainDestroy
```

The difference between text and textId columns is that if an NVTX event message was passed via call to nvtxDomainRegisterString function, then the message will be available through textId field, otherwise the text field will contain the message if it was provided.

**OpenGL Events**

KHR event class values

``` cpp
62 - KhrDebugPushPopRange
63 - KhrDebugGpuPushPopRange
```

KHR source kind values

``` cpp
0x8249 - GL_DEBUG_SOURCE_THIRD_PARTY
0x824A - GL_DEBUG_SOURCE_APPLICATION
```

KHR type values

``` cpp
0x824C - GL_DEBUG_TYPE_ERROR
0x824D - GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR
0x824E - GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR
0x824F - GL_DEBUG_TYPE_PORTABILITY
0x8250 - GL_DEBUG_TYPE_PERFORMANCE
0x8251 - GL_DEBUG_TYPE_OTHER
0x8268 - GL_DEBUG_TYPE_MARKER
0x8269 - GL_DEBUG_TYPE_PUSH_GROUP
0x826A - GL_DEBUG_TYPE_POP_GROUP
```

KHR severity values

``` cpp
0x826B - GL_DEBUG_SEVERITY_NOTIFICATION
0x9146 - GL_DEBUG_SEVERITY_HIGH
0x9147 - GL_DEBUG_SEVERITY_MEDIUM
0x9148 - GL_DEBUG_SEVERITY_LOW
```

**OSRT Event Class Values**

OS runtime libraries can be traced to gather information about low-level userspace APIs. This traces the system call wrappers and thread synchronization interfaces exposed by the C runtime and POSIX Threads (pthread) libraries. This does not perform a complete runtime library API trace, but instead focuses on the functions that can take a long time to execute, or could potentially cause your thread be unscheduled from the CPU while waiting for an event to complete.

OSRT events may have callchains attached to them, depending on selected profiling settings. In such cases, one can use callchainId column to select relevant callchains from OSRT_CALLCHAINS table

OSRT event class values

``` cpp
27 - TRACE_PROCESS_EVENT_OS_RUNTIME
31 - TRACE_PROCESS_EVENT_OS_RUNTIME_START
32 - TRACE_PROCESS_EVENT_OS_RUNTIME_FINISH
```

**DX12 Event Class Values**

``` cpp
41 - TRACE_PROCESS_EVENT_DX12_API
42 - TRACE_PROCESS_EVENT_DX12_WORKLOAD
43 - TRACE_PROCESS_EVENT_DX12_START
44 - TRACE_PROCESS_EVENT_DX12_FINISH
52 - TRACE_PROCESS_EVENT_DX12_DISPLAY
59 - TRACE_PROCESS_EVENT_DX12_CREATE_OBJECT
```

**PIX Event Class Values**

``` cpp
65 - TRACE_PROCESS_EVENT_DX12_DEBUG_API
75 - TRACE_PROCESS_EVENT_DX11_DEBUG_API
```

**Vulkan Event Class Values**

``` cpp
53 - TRACE_PROCESS_EVENT_VULKAN_API
54 - TRACE_PROCESS_EVENT_VULKAN_WORKLOAD
55 - TRACE_PROCESS_EVENT_VULKAN_START
56 - TRACE_PROCESS_EVENT_VULKAN_FINISH
60 - TRACE_PROCESS_EVENT_VULKAN_CREATE_OBJECT
66 - TRACE_PROCESS_EVENT_VULKAN_DEBUG_API
```

**Vulkan Flags**

``` cpp
VALID_BIT = 0x00000001
CACHE_HIT_BIT = 0x00000002
BASE_PIPELINE_ACCELERATION_BIT = 0x00000004
```

**WDDM Event Values**

VIDMM operation type values

``` cpp
0 - None
101 - RestoreSegments
102 - PurgeSegments
103 - CleanupPrimary
104 - AllocatePagingBufferResources
105 - FreePagingBufferResources
106 - ReportVidMmState
107 - RunApertureCoherencyTest
108 - RunUnmapToDummyPageTest
109 - DeferredCommand
110 - SuspendMemorySegmentAccess
111 - ResumeMemorySegmentAccess
112 - EvictAndFlush
113 - CommitVirtualAddressRange
114 - UncommitVirtualAddressRange
115 - DestroyVirtualAddressAllocator
116 - PageInDevice
117 - MapContextAllocation
118 - InitPagingProcessVaSpace
200 - CloseAllocation
202 - ComplexLock
203 - PinAllocation
204 - FlushPendingGpuAccess
205 - UnpinAllocation
206 - MakeResident
207 - Evict
208 - LockInAperture
209 - InitContextAllocation
210 - ReclaimAllocation
211 - DiscardAllocation
212 - SetAllocationPriority
1000 - EvictSystemMemoryOfferList
```

Paging queue type values

``` cpp
0 - VIDMM_PAGING_QUEUE_TYPE_UMD
1 - VIDMM_PAGING_QUEUE_TYPE_Default
2 - VIDMM_PAGING_QUEUE_TYPE_Evict
3 - VIDMM_PAGING_QUEUE_TYPE_Reclaim
```

Packet type values

``` cpp
0 - DXGKETW_RENDER_COMMAND_BUFFER
1 - DXGKETW_DEFERRED_COMMAND_BUFFER
2 - DXGKETW_SYSTEM_COMMAND_BUFFER
3 - DXGKETW_MMIOFLIP_COMMAND_BUFFER
4 - DXGKETW_WAIT_COMMAND_BUFFER
5 - DXGKETW_SIGNAL_COMMAND_BUFFER
6 - DXGKETW_DEVICE_COMMAND_BUFFER
7 - DXGKETW_SOFTWARE_COMMAND_BUFFER
```

Engine type values

``` cpp
0 - DXGK_ENGINE_TYPE_OTHER
1 - DXGK_ENGINE_TYPE_3D
2 - DXGK_ENGINE_TYPE_VIDEO_DECODE
3 - DXGK_ENGINE_TYPE_VIDEO_ENCODE
4 - DXGK_ENGINE_TYPE_VIDEO_PROCESSING
5 - DXGK_ENGINE_TYPE_SCENE_ASSEMBLY
6 - DXGK_ENGINE_TYPE_COPY
7 - DXGK_ENGINE_TYPE_OVERLAY
8 - DXGK_ENGINE_TYPE_CRYPTO
```

DMA interrupt type values

``` cpp
1 = DXGK_INTERRUPT_DMA_COMPLETED
2 = DXGK_INTERRUPT_DMA_PREEMPTED
4 = DXGK_INTERRUPT_DMA_FAULTED
9 = DXGK_INTERRUPT_DMA_PAGE_FAULTED
```

Queue type values

``` cpp
0 = Queue_Packet
1 = Dma_Packet
2 = Paging_Queue_Packet
```

**Driver Events**

Load balance event type values

``` cpp
1 - LoadBalanceEvent_GPU
8 - LoadBalanceEvent_CPU
21 - LoadBalanceMasterEvent_GPU
22 - LoadBalanceMasterEvent_CPU
```

**OpenMP Events**

OpenMP event class values

``` cpp
78 - TRACE_PROCESS_EVENT_OPENMP
79 - TRACE_PROCESS_EVENT_OPENMP_START
80 - TRACE_PROCESS_EVENT_OPENMP_FINISH
```

OpenMP event kind values

``` cpp
15 - OPENMP_EVENT_KIND_TASK_CREATE
16 - OPENMP_EVENT_KIND_TASK_SCHEDULE
17 - OPENMP_EVENT_KIND_CANCEL
20 - OPENMP_EVENT_KIND_MUTEX_RELEASED
21 - OPENMP_EVENT_KIND_LOCK_INIT
22 - OPENMP_EVENT_KIND_LOCK_DESTROY
25 - OPENMP_EVENT_KIND_DISPATCH
26 - OPENMP_EVENT_KIND_FLUSH
27 - OPENMP_EVENT_KIND_THREAD
28 - OPENMP_EVENT_KIND_PARALLEL
29 - OPENMP_EVENT_KIND_SYNC_REGION_WAIT
30 - OPENMP_EVENT_KIND_SYNC_REGION
31 - OPENMP_EVENT_KIND_TASK
32 - OPENMP_EVENT_KIND_MASTER
33 - OPENMP_EVENT_KIND_REDUCTION
34 - OPENMP_EVENT_KIND_MUTEX_WAIT
35 - OPENMP_EVENT_KIND_CRITICAL_SECTION
36 - OPENMP_EVENT_KIND_WORKSHARE
```

OpenMP thread type values

``` cpp
1 - OpenMP Initial Thread
2 - OpenMP Worker Thread
3 - OpenMP Internal Thread
4 - Unknown
```

OpenMP sync region kind values

``` cpp
1 - Barrier
2 - Implicit barrier
3 - Explicit barrier
4 - Implementation-dependent barrier
5 - Taskwait
6 - Taskgroup
```

OpenMP task kind values

``` cpp
1 - Initial task
2 - Implicit task
3 - Explicit task
```

OpenMP prior task status values

``` cpp
1 - Task completed
2 - Task yielded to another task
3 - Task was cancelled
7 - Task was switched out for other reasons
```

OpenMP mutex kind values

``` cpp
1 - Waiting for lock
2 - Testing lock
3 - Waiting for nested lock
4 - Tesing nested lock
5 - Waitng for entering critical section region
6 - Waiting for entering atomic region
7 - Waiting for entering ordered region
```

OpenMP critical section kind values

``` cpp
5 - Critical section region
6 - Atomic region
7 - Ordered region
```

OpenMP workshare kind values

``` cpp
1 - Loop region
2 - Sections region
3 - Single region (executor)
4 - Single region (waiting)
5 - Workshare region
6 - Distrubute region
7 - Taskloop region
```

OpenMP dispatch kind values

``` cpp
1 - Iteration
2 - Section
```

### Common SQLite Examples

**Common Helper Commands**

When utilizing the sqlite3 command line tool, it’s helpful to have data printed as named columns, this can be done with:

``` text
.mode column
.headers on
```

The default column width is determined by the data in the first row of results. If this doesn’t work out well, you can specify widths manually.

``` text
.width 10 20 50
```

**Obtaining Sample Report**

The CLI interface of Nsight Systems was used to profile the radixSortThrust CUDA sample, then the resulting .nsys-rep file was exported using the nsys export.

``` text
nsys profile --trace=cuda,osrt radixSortThrust
nsys export --type sqlite report1.nsys-rep
```

**Serialized Process and Thread Identifiers**

Note

The globalTid field is a 64-bit identifier that encodes multiple components into a single value: \<Hardware ID:8\>\<VM ID:8\>\<Process ID:24\>\<Thread ID:24\> It follows the structure:

- Thread ID: bits 0–23

- Process ID: bits 24–47

- VM ID: bits 48–55

- Hardware ID: bits 56–63

**Goal:** Extract readable process ID (PID) and thread ID (TID) values from Nsight Systems’s serialized identifier format.

Nsight Systems stores identifiers where events originated in serialized form to efficiently pack multiple values into a single field. This example shows how to decode them back to standard PID/TID format.

**What the code does:** Uses bit shifting and modulo operations to extract the embedded PID and TID values from the globalTid field.

For events that have globalTid or globalPid fields exported, use the following code to extract numeric TID and PID.

``` sql
SELECT globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID FROM TABLE_NAME;
```

``` python
# Python equivalent:
def extract_pid_tid(global_tid):
    PID = (global_tid // 0x1000000) % 0x1000000
    TID = global_tid % 0x1000000
    return PID, TID
```

Note

globalTid field includes both TID and PID values, while globalPid only contains the PID value.

**Understanding Event Types and Tags**

Many tables in the nsys SQLite export use numeric codes for event types, tags, and classes. These codes represent different categories of events:

- **Event Types**: Identify the kind of operation or marker (e.g., NVTX marks vs ranges)

- **Tags**: Specify event phases like BEGIN/END for state transitions

- **Event Classes**: Categorize broad event categories (e.g., different types of GPU operations)

When you see numeric values in queries, refer to the documentation sections above or query the relevant string tables to understand their meanings.

**Correlate CUDA Kernel Launches With CUDA API Kernel Launches**

**Goal:** Link CUDA runtime API calls to the actual GPU kernels they launch, enabling analysis of which API calls resulted in the longest-running kernels.

**What the code does:** 1. Adds human-readable columns to the runtime table 2. Joins runtime API calls with GPU kernel executions using correlation IDs 3. Populates kernel names and API function names from the string table 4. Finds the 10 longest API calls that resulted in kernel execution

``` sql
ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN name TEXT;
ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN kernelName TEXT;

UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET kernelName =
    (SELECT value FROM StringIds
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS cuda_gpu
        ON cuda_gpu.shortName = StringIds.id
        AND CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = cuda_gpu.correlationId);

UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name =
    (SELECT value FROM StringIds WHERE nameId = StringIds.id);
```

Select the 10 longest CUDA API ranges that resulted in kernel execution.

``` sql
SELECT name, kernelName, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE kernelName IS NOT NULL ORDER BY end - start LIMIT 10;
```

``` python
# Python equivalent workflow:
# 1. Load runtime and kernel data
# 2. Join on correlation_id to match API calls with kernels
# 3. Add readable names from string table
# 4. Sort by duration and get top 10

runtime_with_kernels = runtime.merge(kernels, on='correlationId')
runtime_with_kernels['duration'] = runtime_with_kernels['end'] - runtime_with_kernels['start']
top_10_longest = runtime_with_kernels.nlargest(10, 'duration')
```

Results:

``` text
name                    kernelName               start       end
----------------------  -----------------------  ----------  ----------
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  658863435   658868490
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  609755015   609760075
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  632683286   632688349
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  606495356   606500439
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  603114486   603119586
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  802729785   802734906
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  593381170   593386294
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  658759955   658765090
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  681549917   681555059
cudaLaunchKernel_v7000  RadixSortScanBinsKernel  717812527   717817671
```

**Remove Ranges Overlapping With Overhead**

**Goal:** Identify and remove CUDA API calls that overlap with profiler overhead to get cleaner performance measurements.

**What the code does:** Uses spatial overlap detection to find CUDA runtime ranges that intersect with profiler overhead periods. The query checks for three types of overlap: range starts within overhead, range ends within overhead, or range completely encompasses overhead.

Use the this query to count CUDA API ranges overlapping with the overhead ones.

Replace “SELECT COUNT(\*)” with “DELETE” to remove such ranges.

``` sql
SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE rowid IN
(
    SELECT cuda.rowid
    FROM PROFILER_OVERHEAD as overhead
    INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME as cuda ON
    (cuda.start BETWEEN overhead.start and overhead.end)
    OR (cuda.end BETWEEN overhead.start and overhead.end)
    OR (cuda.start < overhead.start AND cuda.end > overhead.end)
);
```

``` python
# Python equivalent for finding overlaps:
def ranges_overlap(range1_start, range1_end, range2_start, range2_end):
    return (range1_start <= range2_end and range1_end >= range2_start)

overlapping_ranges = []
for cuda_range in cuda_ranges:
    for overhead_range in overhead_ranges:
        if ranges_overlap(cuda_range.start, cuda_range.end,
                        overhead_range.start, overhead_range.end):
            overlapping_ranges.append(cuda_range)
```

Results:

``` text
COUNT(*)
----------
1095
```

**Find CUDA API Calls that Resulted in the Original Graph Node Creation**

**Goal:** Identify which CUDA API calls were responsible for creating the original nodes in CUDA graphs (as opposed to cloned or instantiated nodes).

**What the code does:** 1. Filters graph nodes to find only original creations (those without originalGraphNodeId) 2. Groups by graphNodeId to get the first occurrence 3. Correlates with CUDA runtime API calls that were active when the graph node was created 4. Joins with string table to get readable API function names

``` sql
SELECT graph.graphNodeId, api.start, graph.start as graphStart, api.end,
    api.globalTid, api.correlationId, api.globalTid,
    (SELECT value FROM StringIds where api.nameId == id) as name
FROM CUPTI_ACTIVITY_KIND_RUNTIME as api
JOIN
    (
        SELECT start, graphNodeId, globalTid from CUDA_GRAPH_NODE_EVENTS
        GROUP BY graphNodeId
        HAVING COUNT(originalGraphNodeId) = 0
    ) as graph
ON api.globalTid == graph.globalTid AND api.start < graph.start AND api.end > graph.start
ORDER BY graphNodeId;
```

Results:

``` text
graphNodeId  start       graphStart  end         globalTid        correlationId  globalTid        name
-----------  ----------  ----------  ----------  ---------------  -------------  ---------------  -----------------------------
1            584366518   584378040   584379102   281560221750233  109            281560221750233  cudaGraphAddMemcpyNode_v10000
2            584379402   584382428   584383139   281560221750233  110            281560221750233  cudaGraphAddMemsetNode_v10000
3            584390663   584395352   584396053   281560221750233  111            281560221750233  cudaGraphAddKernelNode_v10000
4            584396314   584397857   584398438   281560221750233  112            281560221750233  cudaGraphAddMemsetNode_v10000
5            584398759   584400311   584400812   281560221750233  113            281560221750233  cudaGraphAddKernelNode_v10000
6            584401083   584403047   584403527   281560221750233  114            281560221750233  cudaGraphAddMemcpyNode_v10000
7            584403928   584404920   584405491   281560221750233  115            281560221750233  cudaGraphAddHostNode_v10000
29           632107852   632117921   632121407   281560221750233  144            281560221750233  cudaMemcpyAsync_v3020
30           632122168   632125545   632127989   281560221750233  145            281560221750233  cudaMemsetAsync_v3020
31           632131546   632133339   632135584   281560221750233  147            281560221750233  cudaMemsetAsync_v3020
34           632162514   632167393   632169297   281560221750233  151            281560221750233  cudaMemcpyAsync_v3020
35           632170068   632173334   632175388   281560221750233  152            281560221750233  cudaLaunchHostFunc_v10000
```

**Backtraces for OSRT Ranges**

**Goal:** Analyze operating system runtime (OSRT) function calls with their full call stacks to understand performance bottlenecks and call patterns.

**What the code does:** 1. Adds human-readable columns for function names, symbol names, and module names 2. Populates these from the string table for better readability 3. Shows how to query the longest OSRT call with its complete backtrace

Adding text columns makes results of the query below more human-readable.

``` sql
ALTER TABLE OSRT_API ADD COLUMN name TEXT;
UPDATE OSRT_API SET name = (SELECT value FROM StringIds WHERE OSRT_API.nameId = StringIds.id);

ALTER TABLE OSRT_CALLCHAINS ADD COLUMN symbolName TEXT;
UPDATE OSRT_CALLCHAINS SET symbolName = (SELECT value FROM StringIds WHERE symbol = StringIds.id);

ALTER TABLE OSRT_CALLCHAINS ADD COLUMN moduleName TEXT;
UPDATE OSRT_CALLCHAINS SET moduleName = (SELECT value FROM StringIds WHERE module = StringIds.id);
```

Print backtrace of the longest OSRT range.

``` sql
SELECT globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID,
    start, end, name, callchainId, stackDepth, symbolName, moduleName
FROM OSRT_API LEFT JOIN OSRT_CALLCHAINS ON callchainId == OSRT_CALLCHAINS.id
WHERE OSRT_API.rowid IN (SELECT rowid FROM OSRT_API ORDER BY end - start DESC LIMIT 1)
ORDER BY stackDepth LIMIT 10;
```

``` python
# Python equivalent for finding longest call with backtrace:
longest_call = osrt_api.loc[osrt_api['duration'].idxmax()]
backtrace = osrt_callchains[osrt_callchains['id'] == longest_call['callchainId']]
backtrace_ordered = backtrace.sort_values('stackDepth')
```

Results:

``` text
PID         TID         start       end         name                    callchainId  stackDepth  symbolName                      moduleName
----------  ----------  ----------  ----------  ----------------------  -----------  ----------  ------------------------------  ----------------------------------------
19163       19176       360897690   860966851   pthread_cond_timedwait  88           0           pthread_cond_timedwait@GLIBC_2  /lib/x86_64-linux-gnu/libpthread-2.27.so
19163       19176       360897690   860966851   pthread_cond_timedwait  88           1           0x7fbc983b7227                  /usr/lib/x86_64-linux-gnu/libcuda.so.418
19163       19176       360897690   860966851   pthread_cond_timedwait  88           2           0x7fbc9835d5c7                  /usr/lib/x86_64-linux-gnu/libcuda.so.418
19163       19176       360897690   860966851   pthread_cond_timedwait  88           3           0x7fbc983b64a8                  /usr/lib/x86_64-linux-gnu/libcuda.so.418
19163       19176       360897690   860966851   pthread_cond_timedwait  88           4           start_thread                    /lib/x86_64-linux-gnu/libpthread-2.27.so
19163       19176       360897690   860966851   pthread_cond_timedwait  88           5           __clone                         /lib/x86_64-linux-gnu/libc-2.27.so
```

Profiled processes output streams.

**Goal:** Access stdout and stderr output from profiled processes to correlate application output with performance data.

**What the code does:** Resolves file paths and content from string IDs to show the captured stdout/stderr streams from profiled applications.

``` sql
ALTER TABLE ProcessStreams ADD COLUMN filename TEXT;
UPDATE ProcessStreams SET filename = (SELECT value FROM StringIds WHERE ProcessStreams.filenameId = StringIds.id);

ALTER TABLE ProcessStreams ADD COLUMN content TEXT;
UPDATE ProcessStreams SET content = (SELECT value FROM StringIds WHERE ProcessStreams.contentId = StringIds.id);
```

Select all collected stdout and stderr streams.

``` sql
select globalPid / 0x1000000 % 0x1000000 AS PID, filename, content from ProcessStreams;
```

Results:

``` text
PID         filename                                                 content
----------  -------------------------------------------------------  --------------------------------------------------------------------------------------------------------------------
19163       /tmp/nvidia/nsight_systems/streams/pid_19163_stdout.log  /home/user_name/NVIDIA_CUDA-10.1_Samples/6_Advanced/radixSortThrust/radixSortThrust Starting...

GPU Device 0: "Quadro P2000" with compute capability 6.1


Sorting 1048576 32-bit unsigned int keys and values

radixSortThrust, Throughput = 401.0872 MElements/s, Time = 0.00261 s, Size = 1048576 elements
Test passed

19163       /tmp/nvidia/nsight_systems/streams/pid_19163_stderr.log
```

**Thread Summary**

**Goal:** Calculate CPU utilization statistics per thread to identify which threads are consuming the most CPU resources.

Note that Nsight Systems applies additional logic during sampling events processing to work around lost events. This means that the results of the below query might differ slightly from the ones shown in “Analysis Summary” tab.

**Approach 1: Using CPU Cycles (when available)**

**What this code does:** Calculates thread CPU utilization using hardware performance counter data (CPU cycles) which provides the most accurate measurement of actual CPU usage per thread.

``` sql
SELECT
    globalTid / 0x1000000 % 0x1000000 AS PID,
    globalTid % 0x1000000 AS TID,
    ROUND(100.0 * SUM(cpuCycles) /
        (
            SELECT SUM(cpuCycles) FROM COMPOSITE_EVENTS
            GROUP BY globalTid / 0x1000000000000 % 0x100
        ),
        2
    ) as CPU_utilization,
    (SELECT value FROM StringIds WHERE id =
        (
            SELECT nameId FROM ThreadNames
            WHERE ThreadNames.globalTid = COMPOSITE_EVENTS.globalTid
        )
    ) as thread_name
FROM COMPOSITE_EVENTS
GROUP BY globalTid
ORDER BY CPU_utilization DESC
LIMIT 10;
```

Results:

``` text
PID         TID         CPU_utilization  thread_name
----------  ----------  ---------------  ---------------
19163       19163       98.4             radixSortThrust
19163       19168       1.35             CUPTI worker th
19163       19166       0.25             [NS]
```

**Approach 2: Using Scheduling Events (when PMU data not available)**

**What this approach does:** When CPU cycle counter data is not collected, this method calculates thread CPU time based on scheduling events (when threads are scheduled in/out), then calculates utilization percentages. This approach is less precise but still useful for understanding relative thread activity.

``` sql
CREATE INDEX sched_start ON SCHED_EVENTS (start);

CREATE TABLE CPU_USAGE AS
SELECT
    first.globalTid as globalTid,
    (SELECT nameId FROM ThreadNames WHERE ThreadNames.globalTid = first.globalTid) as nameId,
    sum(second.start - first.start) as total_duration,
    count() as ranges_count
FROM SCHED_EVENTS as first
LEFT JOIN SCHED_EVENTS as second
ON second.rowid =
    (
        SELECT rowid
        FROM SCHED_EVENTS
        WHERE start > first.start AND globalTid = first.globalTid
        ORDER BY start ASC
        LIMIT 1
    )
WHERE first.isSchedIn != 0
GROUP BY first.globalTid
ORDER BY total_duration DESC;

SELECT
    globalTid / 0x1000000 % 0x1000000 AS PID,
    globalTid % 0x1000000 AS TID,
    (SELECT value FROM StringIds where nameId == id) as thread_name,
    ROUND(100.0 * total_duration / (SELECT SUM(total_duration) FROM CPU_USAGE), 2) as CPU_utilization
FROM CPU_USAGE
ORDER BY CPU_utilization DESC;
```

``` python
# Python equivalent for scheduling-based calculation:
def calculate_thread_cpu_time(sched_events):
    cpu_usage = {}
    for tid in unique_tids:
        tid_events = sched_events[sched_events['globalTid'] == tid]
        tid_events = tid_events[tid_events['isSchedIn'] == 1]  # Only sched-in events

        total_time = 0
        for i in range(len(tid_events) - 1):
            time_slice = tid_events.iloc[i+1]['start'] - tid_events.iloc[i]['start']
            total_time += time_slice

        cpu_usage[tid] = total_time
    return cpu_usage
```

Results:

``` text
PID         TID         thread_name      CPU_utilization
----------  ----------  ---------------  ---------------
19163       19163       radixSortThrust  93.74
19163       19169       radixSortThrust  3.22
19163       19168       CUPTI worker th  2.46
19163       19166       [NS]             0.44
19163       19172       radixSortThrust  0.07
19163       19167       [NS Comms]       0.05
19163       19176       radixSortThrust  0.02
19163       19170       radixSortThrust  0.0
```

**Function Table**

**Goal:** Create profiler-style function tables showing flat view (total time in each function across all call stacks) and bottom-up view (time spent directly in each function).

**What the code does:** Processes sampling callchain data to calculate time spent in functions, providing two views commonly used in profilers.

These examples demonstrate how to calculate Flat and BottomUp (for top level only) views statistics.

To set up:

``` sql
ALTER TABLE SAMPLING_CALLCHAINS ADD COLUMN symbolName TEXT;
UPDATE SAMPLING_CALLCHAINS SET symbolName = (SELECT value FROM StringIds WHERE symbol = StringIds.id);

ALTER TABLE SAMPLING_CALLCHAINS ADD COLUMN moduleName TEXT;
UPDATE SAMPLING_CALLCHAINS SET moduleName = (SELECT value FROM StringIds WHERE module = StringIds.id);
```

To get flat view:

**Flat view:** Shows total time spent in each function across all call stacks (inclusive time).

``` sql
SELECT symbolName, moduleName, ROUND(100.0 * sum(cpuCycles) /
    (SELECT SUM(cpuCycles) FROM COMPOSITE_EVENTS), 2) AS flatTimePercentage
FROM SAMPLING_CALLCHAINS
LEFT JOIN COMPOSITE_EVENTS ON SAMPLING_CALLCHAINS.id == COMPOSITE_EVENTS.id
GROUP BY symbol, module
ORDER BY flatTimePercentage DESC
LIMIT 5;
```

To get BottomUp view (top level only):

**Bottom-up view:** Shows time spent directly in each function (exclusive time, only leaf functions in call stacks).

``` sql
SELECT symbolName, moduleName, ROUND(100.0 * sum(cpuCycles) /
    (SELECT SUM(cpuCycles) FROM COMPOSITE_EVENTS), 2) AS selfTimePercentage
FROM SAMPLING_CALLCHAINS
LEFT JOIN COMPOSITE_EVENTS ON SAMPLING_CALLCHAINS.id == COMPOSITE_EVENTS.id
WHERE stackDepth == 0
GROUP BY symbol, module
ORDER BY selfTimePercentage DESC
LIMIT 5;
```

``` python
# Python equivalent:
# Flat view - aggregate all occurrences of each function
flat_view = callchains.groupby(['symbol', 'module'])['cpuCycles'].sum()
flat_percentages = (flat_view / total_cycles * 100).sort_values(ascending=False)

# Bottom-up view - only leaf nodes (stackDepth == 0)
leaf_functions = callchains[callchains['stackDepth'] == 0]
bottomup_view = leaf_functions.groupby(['symbol', 'module'])['cpuCycles'].sum()
```

Results:

``` text
symbolName   moduleName   flatTimePercentage
-----------  -----------  ------------------
[Max depth]  [Max depth]  99.92
thrust::zip  /home/user_  24.17
thrust::zip  /home/user_  24.17
thrust::det  /home/user_  24.17
thrust::det  /home/user_  24.17
symbolName      moduleName                                   selfTimePercentage
--------------  -------------------------------------------  ------------------
0x7fbc984982b6  /usr/lib/x86_64-linux-gnu/libcuda.so.418.39  5.29
0x7fbc982d0010  /usr/lib/x86_64-linux-gnu/libcuda.so.418.39  2.81
thrust::iterat  /home/user_name/NVIDIA_CUDA-10.1_Samples/6_  2.23
thrust::iterat  /home/user_name/NVIDIA_CUDA-10.1_Samples/6_  1.55
void thrust::i  /home/user_name/NVIDIA_CUDA-10.1_Samples/6_  1.55
```

**DX12 API Frame Duration Histogram**

**Goal:** Analyze DirectX 12 application frame timing by measuring the duration between consecutive Present calls and creating a timing histogram.

**What the code does:** 1. Creates a view that pairs consecutive Present calls to calculate frame durations 2. Groups frame durations into millisecond buckets 3. Counts how many frames fall into each duration bucket

The example demonstrates how to calculate DX12 CPU frames durartion and construct a histogram out of it.

``` sql
CREATE INDEX DX12_API_ENDTS ON DX12_API (end);

CREATE TEMP VIEW DX12_API_FPS AS SELECT end AS start,
    (SELECT end FROM DX12_API
        WHERE end > outer.end AND nameId == (SELECT id FROM StringIds
            WHERE value == "IDXGISwapChain::Present")
        ORDER BY end ASC LIMIT 1) AS end
FROM DX12_API AS outer
    WHERE nameId == (SELECT id FROM StringIds WHERE value == "IDXGISwapChain::Present")
ORDER BY end;
```

Number of frames with a duration of \[X, X + 1\] milliseconds.

``` sql
SELECT
    CAST((end - start) / 1000000.0 AS INT) AS duration_ms,
    count(*)
FROM DX12_API_FPS
WHERE end IS NOT NULL
GROUP BY duration_ms
ORDER BY duration_ms;
```

``` python
# Python equivalent:
present_calls = dx12_api[dx12_api['function_name'] == 'IDXGISwapChain::Present']
present_calls = present_calls.sort_values('end')

frame_durations = []
for i in range(len(present_calls) - 1):
    duration = present_calls.iloc[i+1]['end'] - present_calls.iloc[i]['end']
    duration_ms = duration / 1000000.0  # Convert to milliseconds
    frame_durations.append(int(duration_ms))

# Create histogram
histogram = pd.Series(frame_durations).value_counts().sort_index()
```

Results:

``` text
duration_ms  count(*)
-----------  ----------
3            1
4            2
5            7
6            153
7            19
8            116
9            16
10           8
11           2
12           2
13           1
14           4
16           3
17           2
18           1
```

**GPU Context Switch Events Enumeration**

**Goal:** Track GPU context switches to understand GPU scheduling behavior and identify context switch patterns.

**What the code does:** Filters GPU context switch events to show only BEGIN (tag=8) and END (tag=7) events, which mark the boundaries of GPU context execution periods.

**GPU Context Switch Event Tags:** - **7**: END events (context execution ends) - **8**: BEGIN events (context execution begins)

GPU context duration is between first BEGIN and a matching END event.

``` sql
SELECT (CASE tag WHEN 8 THEN "BEGIN" WHEN 7 THEN "END" END) AS tag,
    globalPid / 0x1000000 % 0x1000000 AS PID,
    vmId, seqNo, contextId, timestamp, gpuId FROM GPU_CONTEXT_SWITCH_EVENTS
WHERE tag in (7, 8) ORDER BY seqNo LIMIT 10;
```

Results:

``` text
tag         PID         vmId        seqNo       contextId   timestamp   gpuId
----------  ----------  ----------  ----------  ----------  ----------  ----------
BEGIN       23371       0           0           1048578     56759171    0
BEGIN       23371       0           1           1048578     56927765    0
BEGIN       23371       0           3           1048578     63799379    0
END         23371       0           4           1048578     63918806    0
BEGIN       19397       0           5           1048577     64014692    0
BEGIN       19397       0           6           1048577     64250369    0
BEGIN       19397       0           8           1048577     1918310004  0
END         19397       0           9           1048577     1918521098  0
BEGIN       19397       0           10          1048577     2024164744  0
BEGIN       19397       0           11          1048577     2024358650  0
```

**Resolve NVTX Category Name**

**Goal:** Decode NVTX category names for NVTX markers and ranges to make the profiling data more human-readable.

**What the code does:** Joins NVTX events with their category definitions to resolve category IDs into meaningful category names, making it easier to understand the purpose of different NVTX annotations.

**NVTX Event Types:**

> - **33**: Category definition events (define new categories)
>
> - **34**: Mark events (instantaneous markers)
>
> - **59**: Push/Pop range events (nested ranges)
>
> - **60**: Start/End range events (paired ranges)

The example demonstrates how to resolve NVTX category name for NVTX marks and ranges.

``` sql
WITH
  event AS (
    SELECT *
    FROM NVTX_EVENTS
    WHERE eventType IN (34, 59, 60) -- mark, push/pop, start/end
  ),
  category AS (
    SELECT
      category,
      domainId,
      text AS categoryName
    FROM NVTX_EVENTS
    WHERE eventType == 33 -- category definition events
  )
SELECT
  start,
  end,
  globalTid,
  eventType,
  domainId,
  category,
  categoryName,
  text
FROM event JOIN category USING (category, domainId)
ORDER BY start;
```

Results:

``` text
start       end         globalTid        eventType   domainId    category    categoryName               text
----------  ----------  ---------------  ----------  ----------  ----------  -------------------------  ----------------
18281150    18311960    281534938484214  59          0           1           FirstCategoryUnderDefault  Push Pop Range A
18288187    18306674    281534938484214  59          0           2           SecondCategoryUnderDefaul  Push Pop Range B
18294247                281534938484214  34          0           1           FirstCategoryUnderDefault  Mark A
18300034                281534938484214  34          0           2           SecondCategoryUnderDefaul  Mark B
18345546    18372595    281534938484214  60          1           1           FirstCategoryUnderMyDomai  Start End Range
18352924    18378342    281534938484214  60          1           2           SecondCategoryUnderMyDoma  Start End Range
18359634                281534938484214  34          1           1           FirstCategoryUnderMyDomai  Mark A
18365448                281534938484214  34          1           2           SecondCategoryUnderMyDoma  Mark B
```

**Rename CUDA Kernels with NVTX**

**Goal:** Associate CUDA kernels with their surrounding NVTX ranges to provide more meaningful names and context for kernel analysis.

**What the code does:** 1. Finds the innermost NVTX push-pop range that encompasses each CUDA kernel launch 2. Maps the NVTX range text to the corresponding kernel execution 3. Enables analysis of kernels by their logical function rather than just their technical names

The example demonstrates how to map innermost NVTX push-pop range to a matching CUDA kernel run.

``` sql
ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN nvtxRange TEXT;
CREATE INDEX nvtx_start ON NVTX_EVENTS (start);


UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET nvtxRange = (
    SELECT NVTX_EVENTS.text
    FROM NVTX_EVENTS JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON
        NVTX_EVENTS.eventType == 59 AND
        NVTX_EVENTS.globalTid == CUPTI_ACTIVITY_KIND_RUNTIME.globalTid AND
        NVTX_EVENTS.start <= CUPTI_ACTIVITY_KIND_RUNTIME.start AND
        NVTX_EVENTS.end >= CUPTI_ACTIVITY_KIND_RUNTIME.end
    WHERE
        CUPTI_ACTIVITY_KIND_KERNEL.correlationId == CUPTI_ACTIVITY_KIND_RUNTIME.correlationId
    ORDER BY NVTX_EVENTS.start DESC LIMIT 1
);

SELECT start, end, globalPid, StringIds.value as shortName, nvtxRange
FROM CUPTI_ACTIVITY_KIND_KERNEL JOIN StringIds ON shortName == id
ORDER BY start LIMIT 6;
```

``` python
# Python equivalent:
def find_innermost_nvtx_range(kernel_start, kernel_end, nvtx_ranges):
    # Find NVTX ranges that completely contain the kernel
    containing_ranges = []
    for nvtx in nvtx_ranges:
        if nvtx['start'] <= kernel_start and nvtx['end'] >= kernel_end:
            containing_ranges.append(nvtx)

    # Return the innermost (latest starting) range
    if containing_ranges:
        return max(containing_ranges, key=lambda x: x['start'])['text']
    return None
```

Results:

``` text
start       end         globalPid          shortName      nvtxRange
----------  ----------  -----------------  -------------  ----------
526545376   526676256   72057700439031808  MatrixMulCUDA
526899648   527030368   72057700439031808  MatrixMulCUDA  Add
527031648   527162272   72057700439031808  MatrixMulCUDA  Add
527163584   527294176   72057700439031808  MatrixMulCUDA  My Kernel
527296160   527426592   72057700439031808  MatrixMulCUDA  My Range
527428096   527558656   72057700439031808  MatrixMulCUDA
```

**Select CUDA Calls With Backtraces**

**Goal:** Analyze CUDA API calls along with their call stacks to understand the application code paths that lead to CUDA API usage.

**What the code does:** Joins CUDA runtime API calls with their associated call chains to show the complete stack trace for each CUDA call, helping identify where in the application CUDA calls originate.

``` sql
ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN name TEXT;
UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name = (SELECT value FROM StringIds WHERE CUPTI_ACTIVITY_KIND_RUNTIME.nameId = StringIds.id);

ALTER TABLE CUDA_CALLCHAINS ADD COLUMN symbolName TEXT;
UPDATE CUDA_CALLCHAINS SET symbolName = (SELECT value FROM StringIds WHERE symbol = StringIds.id);

SELECT globalTid % 0x1000000 AS TID,
    start, end, name, callchainId, stackDepth, symbolName
FROM CUDA_CALLCHAINS JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON callchainId == CUDA_CALLCHAINS.id
ORDER BY callchainId, stackDepth LIMIT 11;
```

Results:

``` text
TID         start       end         name           callchainId  stackDepth  symbolName
----------  ----------  ----------  -------------  -----------  ----------  --------------
11928       168976467   169077826   cuMemAlloc_v2  1            0           0x7f13c44f02ab
11928       168976467   169077826   cuMemAlloc_v2  1            1           0x7f13c44f0b8f
11928       168976467   169077826   cuMemAlloc_v2  1            2           0x7f13c44f3719
11928       168976467   169077826   cuMemAlloc_v2  1            3           cuMemAlloc_v2
11928       168976467   169077826   cuMemAlloc_v2  1            4           cudart::driver
11928       168976467   169077826   cuMemAlloc_v2  1            5           cudart::cudaAp
11928       168976467   169077826   cuMemAlloc_v2  1            6           cudaMalloc
11928       168976467   169077826   cuMemAlloc_v2  1            7           cudaError cuda
11928       168976467   169077826   cuMemAlloc_v2  1            8           main
11928       168976467   169077826   cuMemAlloc_v2  1            9           __libc_start_m
11928       168976467   169077826   cuMemAlloc_v2  1            10          _start
```

**SLI Peer-to-Peer Query**

**Goal:** Filter and analyze SLI (Scalable Link Interface) peer-to-peer memory transfers between GPUs based on size, timing, and other criteria.

**What the code does:** Demonstrates how to query SLI P2P events with filtering conditions on resource size, time range, and sorting by transfer size to identify significant GPU-to-GPU transfers.

**SLI P2P Event Classes:** - **62**: Peer-to-peer transfer events between GPUs

The example demonstrates how to query SLI Peer-to-Peer events with resource size greater than value and within a time range sorted by resource size descending.

``` sql
SELECT *
FROM SLI_P2P
WHERE resourceSize < 98304 AND start > 1568063100 AND end < 1579468901
ORDER BY resourceSize DESC;
```

``` python
# Python equivalent:
filtered_transfers = sli_p2p[
    (sli_p2p['resourceSize'] < 98304) &
    (sli_p2p['start'] > 1568063100) &
    (sli_p2p['end'] < 1579468901)
].sort_values('resourceSize', ascending=False)
```

Results:

``` text
start       end         eventClass  globalTid          gpu         frameId     transferSkipped  srcGpu      dstGpu      numSubResources  resourceSize  subResourceIdx  smplWidth   smplHeight  smplDepth   bytesPerElement  dxgiFormat  logSurfaceNames  transferInfo  isEarlyPushManagedByNvApi  useAsyncP2pForResolve  transferFuncName  regimeName  debugName   bindType
----------  ----------  ----------  -----------------  ----------  ----------  ---------------  ----------  ----------  ---------------  ------------  --------------  ----------  ----------  ----------  ---------------  ----------  ---------------  ------------  -------------------------  ---------------------  ----------------  ----------  ----------  ----------
1570351100  1570351101  62          72057698056667136  0           771         0                256         512         1                1048576       0               256         256         1           16               2                            3             0                          0
1570379300  1570379301  62          72057698056667136  0           771         0                256         512         1                1048576       0               64          64          64          4                31                           3             0                          0
1572316400  1572316401  62          72057698056667136  0           773         0                256         512         1                1048576       0               256         256         1           16               2                            3             0                          0
1572345400  1572345401  62          72057698056667136  0           773         0                256         512         1                1048576       0               64          64          64          4                31                           3             0                          0
1574734300  1574734301  62          72057698056667136  0           775         0                256         512         1                1048576       0               256         256         1           16               2                            3             0                          0
1574767200  1574767201  62          72057698056667136  0           775         0                256         512         1                1048576       0               64          64          64          4                31                           3             0                          0
```

**Generic Events**

**Goal:** Analyze system-level events captured through generic event collection (like ftrace) to understand system behavior and syscall patterns.

**What the code does:** Demonstrates how to query generic events stored in JSON format, specifically showing how to create a histogram of syscall usage by process ID. The query uses a subquery to find the specific event type ID for “raw_syscalls:sys_enter” events, then counts occurrences by process ID.

Syscall usage histogram by PID:

``` sql
SELECT json_extract(data, '$.common_pid') AS PID, count(*) AS total
FROM GENERIC_EVENTS WHERE PID IS NOT NULL AND typeId = (
  SELECT typeId FROM GENERIC_EVENT_TYPES
  WHERE json_extract(data, '$.Name') = "raw_syscalls:sys_enter")
GROUP BY PID
ORDER BY total DESC
LIMIT 10;
```

``` python
# Python equivalent:
import json

# Filter for syscall enter events
syscall_events = []
for event in generic_events:
    data = json.loads(event['data'])
    if 'common_pid' in data:
        syscall_events.append(data['common_pid'])

# Count syscalls by PID
pid_counts = pd.Series(syscall_events).value_counts().head(10)
```

Results:

``` text
PID         total
----------  ----------
5551        32811
9680        3988
4328        1477
9564        1246
4376        1204
4377        1167
4357        656
4355        655
4356        640
4354        633
```

**Fetching Generic Events in JSON Format**

**Goal:** Export generic events, types, and sources in JSON format for external analysis tools or custom processing pipelines.

**What the code does:** Constructs JSON objects from the database tables containing generic event data, enabling export to JSON Lines format for further processing with external tools.

Text and JSON export modes don’t include generic events. Use the below queries (without the LIMIT clause) to extract JSON lines representation of generic events, types, and sources.

``` sql
SELECT json_insert('{}',
    '$.sourceId', sourceId,
    '$.data', json(data)
)
FROM GENERIC_EVENT_SOURCES LIMIT 2;

SELECT json_insert('{}',
    '$.typeId', typeId,
    '$.sourceId', sourceId,
    '$.data', json(data)
)
FROM GENERIC_EVENT_TYPES LIMIT 2;

SELECT json_insert('{}',
    '$.rawTimestamp', rawTimestamp,
    '$.timestamp', timestamp,
    '$.typeId', typeId,
    '$.data', json(data)
)
FROM GENERIC_EVENTS LIMIT 2;
```

Results:

``` text
json_insert('{}',
    '$.sourceId', sourceId,
    '$.data', json(data)
)
---------------------------------------------------------------------------------------------------------------
{"sourceId":72057602627862528,"data":{"Name":"FTrace","TimeSource":"ClockMonotonicRaw","SourceGroup":"FTrace"}}
json_insert('{}',
    '$.typeId', typeId,
    '$.sourceId', sourceId,
    '$.data', json(data)
)
--------------------------------------------------------------------------------------------------------------------
{"typeId":72057602627862547,"sourceId":72057602627862528,"data":{"Name":"raw_syscalls:sys_enter","Format":"\"NR %ld (%lx, %lx, %lx, %lx, %lx, %lx)\", REC->id, REC->args[0], REC->args[1], REC->args[2], REC->args[3], REC->args[4], REC->args[5]","Fields":[{"Name":"common_pid","Prefix":"int","Suffix":""},{"Name":"id","Prefix":"long","S
{"typeId":72057602627862670,"sourceId":72057602627862528,"data":{"Name":"irq:irq_handler_entry","Format":"\"irq=%d name=%s\", REC->irq, __get_str(name)","Fields":[{"Name":"common_pid","Prefix":"int","Suffix":""},{"Name":"irq","Prefix":"int","Suffix":""},{"Name":"name","Prefix":"__data_loc char[]","Suffix":""},{"Name":"common_type",
json_insert('{}',
    '$.rawTimestamp', rawTimestamp,
    '$.timestamp', timestamp,
    '$.typeId', typeId,
    '$.data', json(data)
)
--------------------------------------------------------------------------------------------------------------------
{"rawTimestamp":1183694330725221,"timestamp":6236683,"typeId":72057602627862670,"data":{"common_pid":"0","irq":"66","name":"327696","common_type":"142","common_flags":"9","common_preempt_count":"0"}}
{"rawTimestamp":1183694333695687,"timestamp":9207149,"typeId":72057602627862670,"data":{"common_pid":"0","irq":"66","name":"327696","common_type":"142","common_flags":"9","common_preempt_count":"0"}}
```

### Arrow

The Arrow type exported file, `.arrows`, uses the IPC stream format to store all tables in a file. The tables can be read by opening the file as an arrow stream. For example one can use the `open_stream` function from the arrow python package. For more information on the interfaces that can be used to read an IPC stream file, please refer to the Apache Arrow documentation \[[1](https://arrow.apache.org/docs/python/api/ipc.html), [2](https://arrow.apache.org/docs/python/ipc.html)\].

The name of each table is included in the schema metadata. Thus, while reading each table, the user can extract the table title from the metadata. The table name metadata field has the key `table_name`. The titles of all the available tables can be found in section [SQLite Schema Reference](index.html#sqlite-schema-reference).

A sample function that reads all Arrow tables in a `.arrows` file is provided below in Python:

``` python
import pyarrow as pa

def read_tables(arrow_file):
    with pa.input_stream(arrow_file) as source:
        while source.tell() < source.size():
            try:
                yield pa.ipc.open_stream(arrow_file).read_all()
            except:
                continue
```

The Arrow directory exporter type, `_arwdir`, will create a directory with one arrow file per table/dataset.

### JSON Lines

In the JSON Lines export format ([JSON Lines Documentation](https://jsonlines.org/)), events and other report data (such as strings and processes) are serialized into JSON objects, with each object written to a new line.

Output layout:

``` text
{"id":0,"table":"StringIds","value":"[Unknown]"}
{"globalPid":284057963331584,"name":"chrome","pid":153958,"table":"PROCESSES"}
{"globalTid":281523009882942,"nameId":442,"priority":20,"table":"ThreadNames"}
{"name":"COLLECT_GPU_CTX_SW_TRACE","table":"META_DATA_CAPTURE","value":"false"}
...
```

Note the presence of the “table” field in each JSON object. This field allows readers to identify the type of the event and corresponds to the table name in the `sqlite` export.


---


<!-- source: https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html -->
# Installation Guide

NVIDIA Nsight Systems installation guide.

## Overview

Nsight Systems is a statistical sampling profiler with tracing features. It is designed to work with devices and devkits based on NVIDIA Tegra SoCs (system-on-chip), Arm SBSA (server based system architecture) systems, and systems based on the x86_64 processor architecture that also include NVIDIA GPU(s).

Throughout this document we will refer to the device on which profiling happens as the **target**, and the computer on which the user works and controls the profiling session as the **host**. Note that for x86_64 based systems these may be on the same device, whereas with Tegra or Arm based systems they will always be separate.

Furthermore, three different activities are distinguished as follows:

- **Profiling** — The process of collecting any performance data. A profiling session in Nsight Systems typically includes sampling and tracing.

- **Sampling** — The process of periodically stopping the *profilee* (the application under investigation during the profiling session), typically to collect backtraces (call stacks of active threads), which allows you to understand statistically how much time is spent in each function. Additionally, hardware counters can also be sampled. This process is inherently imprecise when a low number of samples have been collected.

- **Tracing** — The process of collecting precise information about various activities happening in the profilee or in the system. For example, profilee API execution may be traced providing the exact time and duration of a function call.

Nsight Systems supports multiple generations of Tegra SoCs, NVIDIA discrete GPUs, and various CPU architectures, as well as various target and host operating systems. This documentation describes the full set of features available in any version of Nsight Systems. In the event that a feature is not available in all versions, that will be noted in the text. In general, Nsight Systems Embedded Platforms Edition indicates the package that supports Tegra processors for the embedded and automotive market and Nsight Systems Workstation Edition supports x86_64 and Arm server (SBSA) processors for the workstation, cluster, and cloud markets.

Common features that are supported by Nsight Systems on most platforms include the following:

- Sampling of the profilee and collecting backtraces using multiple algorithms (such as frame pointers or DWARF data). Building top-down, bottom-up, and flat views as appropriate. This information helps identify performance bottlenecks in CPU-intensive code.

- Sampling or tracing system power behaviors, such as CPU frequency.

- (Only on Nsight Systems Embedded Platforms Edition)Sampling counters from Arm PMU (Performance Monitoring Unit). Information such as cache misses gets statistically correlated with function execution.

- Support for multiple windows. Users with multiple monitors can see multiple reports simultaneously, or have multiple views into the same report file.

With Nsight Systems, a user could:

- Identify call paths that monopolize the CPU.

- Identify individual functions that monopolize the CPU (across different call paths).

- For Nsight Systems Embedded Platforms Edition, identify functions that have poor cache utilization.

- If platform supports CUDA, see visual representation of CUDA Runtime and Driver API calls, as well as CUDA GPU workload. Nsight Systems uses the CUDA Profiling Tools Interface (CUPTI), for more information, see: [CUPTI documentation](https://docs.nvidia.com/cuda/cupti/index.html).

- If the user annotates with NVIDIA Tools Extension (NVTX), see visual representation of NVTX annotations: ranges, markers, and thread names.

- For Windows targets, see visual representation of D3D12: which API calls are being made on the CPU, graphic frames, stutter analysis, as well as GPU workloads (command lists and debug ranges).

- For x86_64 targets, see visual representation of Vulkan: which API calls are being made on the CPU, graphic frames, stutter analysis, as well as Vulkan GPU workloads (command buffers and debug ranges).

## System Requirements

Nsight Systems supports multiple platforms. For simplicity, think of these as Nsight Systems Embedded Platforms Edition and Nsight Systems Workstation Edition, where Nsight Systems Workstation Edition supports desktops, workstations, clusters, and clouds with x86_64 or Arm SBSA CPUs on Linux and Windows OSs, while Nsight Systems Embedded Platforms Edition supports NVIDIA Tegra products for the embedded and gaming space on Linux for Tegra and QNX OSs.

### Supported Platforms

Depending on your OS, different GPUs are supported

L4T (Linux for Tegra)

- Based on your Jetson version, select the appropriate JetPack

- For current Jetson targets download [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack).

- For older Tegra targets, see [NVIDIA JetPack Archives](https://developer.nvidia.com/embedded/jetpack-archive).

x86_64 or Arm SBSA

- NVIDIA GPU architectures starting with Turing

- OS (64 bit only)

  - Ubuntu 22.04, 24.04, and 26.04

  - CentOS 8.0 and RedHat Enterprise Linux 8+

  - Amazon Linux 2023+

  - Win Server 2022+

  - MacOS 13+

Networking Components

- NVIDIA DPUs

- NVIDIA SuperNICs

- Amazon EFA NICs

### CUDA Version

- Nsight Systems supports CUDA 10.0+ for most platforms

- Nsight Systems on Arm SBSA supports 10.2+

Note that CUDA version and driver version must be compatible.

| CUDA Version | Driver minimum version |
|--------------|------------------------|
| 11.0         | 450                    |
| 10.2         | 440.30                 |
| 10.1         | 418.39                 |
| 10.0         | 410.48                 |

From CUDA 11.X on, any driver from 450 on will be supported, although new features introduced in more recent drivers will not be available.

For information about which drivers were specifically released with each toolkit, see [CUDA Toolkit Release Notes - Major Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)

### Requirements for x86_64 and Arm SBSA Targets on Linux

When attaching to x86_64 or Arm SBSA Linux-based target from the GUI on the host, the connection is established through SSH.

**Use of Linux Perf**: To collect thread scheduling data and IP (instruction pointer) samples, the Linux operating system’s `perf_event_paranoid` level must be 2 or less. Use the following command to check:

> ``` text
> cat /proc/sys/kernel/perf_event_paranoid
> ```

If the output is \>2, then do the following to temporarily adjust the paranoid level (note that this has to be done after each reboot):

> ``` text
> sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
> ```

To make the change permanent, use the following command:

> ``` text
> sudo sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'
> ```

**Kernel version**: To collect thread scheduling data and IP (instruction pointer) samples and backtraces, the kernel version must be:

- 3.10.0-693 or later for CentOS and RedHat Enterprise Linux 7.4+

- 4.3 or greater for all other distros including Ubuntu

To check the version number of the kernel on a target device, run the following command on the device:

> ``` text
> uname -a
> ```

Note

Only CentOS, RedHat, and Ubuntu distros are tested/confirmed to work correctly.

**glibc version**: To check the glibc version on a target device, run the following command:

> ``` text
> ldd --version
> ```

Nsight Systems requires glibc 2.17 or newer.

**CUDA**: See above for supported CUDA versions in this release. Use the deviceQuery command to determine the CUDA driver and runtime versions on the system. The deviceQuery command is available in the CUDA SDK. It is normally installed at:

> ``` text
> /usr/local/cuda/samples/1_Utilities/deviceQuery
> ```

Only pure 64-bit environments are supported. In other words, 32-bit systems or 32-bit processes running within a 64-bit environment are not supported.

Nsight Systems requires write permission to the `/var/lock` directory on the target system.

**Docker**: See Container Support section of the User Guide for more information.

### Requirements for x86_64 Targets on Windows

**DX12 Requires**:

- Windows 10 with NVIDIA Driver 411.63 or higher for DX12 trace

- Windows 10 April 2018 Update (version 1803, AKA Redstone 4) with NVIDIA Driver 411.63 or higher for DirectX Ray Tracing, and tracing DX12 Copy command queues.

### Requirements for QNX targets

**Development environment**:

Nsight Systems supports profiling DRIVE OS QNX targets in development environments.

Some features require additional setup, such as deploying specific files to the target system or creating configuration files. See [Profiling with DRIVE Hypervisor](../UserGuide/index.html#profiling-embedded-virtual-machines) for detailed instructions.

**Safety environment**:

Nsight Systems provides limited profiling capabilities in QNX Safety environment.

Warning

Nsight Systems is a profiling and analysis tool that is not safety-certified. It must not be used in environments where software controls driving decisions or impacts human safety.

The *prod_debug_extra* overlay is required to enable Nsight Systems in safety environment.

Available features:

| Feature name | First supported in |
|----|----|
| Tracelogger trace (CPU thread states and context switches) | 6.0.8.x |
| Hypervisor trace (VM context switches, interrupts, traps, etc. - collected through eventlib) | 6.0.8.x |
| VMProfiler (Cross-Hypervisor sampling) | 6.0.8.x |
| OSRT trace (trace of C runtime functions) | 6.0.8.x |
| NVTX trace (trace of user-added NVTX instrumentation) | 6.0.8.x |

Note

For DRIVE OS installation details, more information on running environments or filesystem overlays, please refer to [NVIDIA DRIVE OS Documentation](https://docs.nvidia.com/drive).

### Host Application Requirements

The Nsight Systems host application runs on the following host platforms:

- Windows 10, Windows Server 2019. Only 64-bit versions are supported.

- Linux Ubuntu 14.04 and higher are known to work, running on other modern distributions should be possible as well. Only 64-bit versions are supported.

- OS X 10.10 “Yosemite” and higher.

## Getting Started Guide

### Finding the Right Package

Nsight Systems is available for multiple targets and multiple host OSs. To choose the right package, first consider the target system to be analyzed.

- For Tegra target systems, select Nsight Systems Embedded Platforms Edition available as part of [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack). For older Tegra targets, see [NVIDIA JetPack Archives](https://developer.nvidia.com/embedded/jetpack-archive).

- For x86_64 or Arm SBSA select from the target packages from Nsight Systems Workstation Edition, available from [https://developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems). This web release will always contain the latest and greatest Nsight Systems features.

- The x86_64 and Arm SBSA target versions of Nsight Systems are also available in the [CUDA Toolkit.](https://developer.nvidia.com/cuda-downloads)

Each package is limited to one architecture. For example, Tegra packages do not contain support for profiling x86 targets, and x86 packages do not contain support for profiling Tegra targets.

After choosing an appropriate target version, select the package corresponding to the host OS, the OS on the system where results will be viewed. These packages are in the form of common installer types: .msi for Windows; .run, .rpm, and .deb for x86 Linux; and .dmg for the macOS installer.

**Tegra packages**

- Windows host - Install .msi on Windows machine. Enables remote access to Tegra device for profiling.

- Linux host - Install .run on Linux system. Enables remote access to Tegra device for profiling.

- macOS host - Install .dmg on macOS machine. Enables remote access to Tegra device for profiling.

**x86_64 packages**

- Windows host - Install .msi on Windows machine. Enables remote access to Linux x86_64 or Windows devices for profiling as well as running on local system.

- Linux host - Install .run, .rpm, or .deb on Linux system. Enables remote access to Linux x86_64 or Windows devices for profiling or running collection on localhost.

- Linux CLI only - The Linux CLI is shipped in all x86 packages, but if you just want the CLI, we have a package for that. Install .deb or .rpm on Linux system. Enables only CLI collection, report can be imported or opened in x86_64 host.

- macOS host - Install .dmg on macOS machine. Enables remote access to Linux x86_64 device for profiling.

**Arm SBSA packages**

- Arm SBSA host - Install .run, .rpm, or .deb on Arm SBSA system. Enables profiling and report viewing on local system.

- Arm SBSA CLI only - The Arm SBSA CLI is shipped in all host packages, but if you just want the CLI, we have a package for that. Install .deb or .rpm on Arm SBSA system. Enables only CLI collection, report can be imported or opened in GUI on any supported host platform.

Note

On Windows machines we recommend installing Nsight Systems to the default secure location under Program Files.

### Package Manager Installation

Installation using RPM or Debian packages interfaces with your system’s package management system. When using RPM or Debian local repo installers, the downloaded package contains a repository snapshot stored on the local filesystem in /var/. Such a package only informs the package manager where to find the actual installation packages, but will not install them.

If the online network repository is enabled, RPM or Debian packages will be automatically downloaded at installation time using the package manager: apt-get, dnf, yum, or zypper.

Users can download Nsight Systems (full package **nsight-systems** or CLI-only package **nsight-systems-cli**) from publicly available repositories. The below commands are given as examples and are not intended to be precisely correct.

**Ubuntu (minimal setup for containers)**

These instructions assume that you have root in the container. Example command to launch a container: `sudo`` ``docker`` ``run`` ``-it`` ``--rm`` ``ubuntu:latest`` ``bash`

``` text
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

**Ubuntu (desktop)**

``` text
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install nsight-systems
```

**CentOS and RHEL (minimal setup for containers)**

Same as above for Ubuntu, these instructions assume that you have root in the container. Example command to launch a container: `sudo`` ``docker`` ``run`` ``-it`` ``--rm`` ``centos:latest`` ``bash`

``` text
rpm --import https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
dnf install -y 'dnf-command(config-manager)'
dnf config-manager --add-repo "https://developer.download.nvidia.com/devtools/repos/rhel$(source /etc/os-release; echo ${VERSION_ID%%.*})/$(rpm --eval '%{_arch}' | sed s/aarch/arm/)/"
dnf install -y nsight-systems-cli
```

**CentOS and RHEL (desktop)**

``` text
sudo rpm --import https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo "https://developer.download.nvidia.com/devtools/repos/rhel$(source /etc/os-release; echo ${VERSION_ID%%.*})/$(rpm --eval '%{_arch}' | sed s/aarch/arm/)/"
sudo dnf install nsight-systems
```

### Installing GUI on the Host System

Copy the appropriate file to your host system in a directory where you have write and execute permissions. Run the install file, accept the EULA, and Nsight Systems will install on your system.

On Linux, there are special options to enable automated installation. Running the installer with the `--accept` flag will automatically accept the EULA, running with the `--accept` flag and the `--quiet` flag will automatically accept the EULA without printing to stdout. Running with `--quiet` without `--accept` will display an error.

The installation will create a Host directory for this host and a Target directory for each target this Nsight Systems package supports.

All binaries needed to collect data on a target device will be installed on the target by the host on first connection to the device. There is no need to install the package on the target device.

If installing from the CUDA Toolkit, see the [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/).

### Optional: Setting up the CLI

All Nsight Systems targets can be profiled using the CLI. Arm SBSA targets can only be profiled using the CLI. The CLI is especially helpful when scripts are used to run unattended collections or when access to the target system via ssh is not possible. In particular, this can be used to enable collection in a Docker container.

The CLI can be found in the Target directory of the Nsight Systems installation. Users who want to install the CLI as a standalone tool can do so by copying the files within the Target directory to the location of their choice.

If you wish to run the CLI without root (recommended mode) you will want to install in a directory where you have full access.

Once you have the CLI set up, you can use the `nsys`` ``status`` ``-e` command to check your environment.

``` cpp
~$ nsys status -e

Sampling Environment Check
Linux Kernel Paranoid Level = 1: OK
Linux Distribution = Ubuntu
Linux Kernel Version = 4.15.0-109-generic: OK
Linux perf_event_open syscall available: OK
Sampling trigger event available: OK
Intel(c) Last Branch Record support: Available
Sampling Environment: OK
```

This status check allows you to ensure that the system requirements for CPU sampling using Nsight Systems are met in your local environment. If the Sampling Environment is not OK, you will still be able to run various trace operations.

Intel(c) Last Branch Record allows tools, including Nsight Systems to use hardware to quickly get limited stack information. Nsight Systems will use this method for stack resolution by default if available.

For information about changing these environment settings, see System Requirements section in the Installation Guide. For information about changing the backtrace method, see Profiling from the CLI in the User Guide.

To get started using the CLI, run `nsys`` ``--help` for a list of options or see Profiling Applications from the CLI in the User Guide for full documentation.

### Launching the GUI

Depending on your OS, Nsight Systems will have installed an icon on your host desktop that you can use to launch the GUI. To launch the GUI directly, run the `nsys-ui` executable in the Host sub-directory of your installation.

### Installing Advanced Analysis System

The Nsight Systems advanced analysis system can be located in the `<install-dir>/target-linux-x64/python/packages` directory.

**Recipe Dependencies**

The system is written in Python and depends on a set of Python packages. The prerequisites are Python 3.10 or newer with pip and venv. If you don’t have Python, you can install it from [python.org](https://www.python.org) or the Windows store or Linux package manager.

The recipe system is supported on the currently maintained Python releases, as shown in [this chart](https://www.python.org/downloads/). We are not able to develop and test the recipe system with older Python versions that are no longer supported by Python.

**Pip/venv on Ubuntu**

If pip/venv were not installed with Python, run:

``` text
$ sudo apt-get install python3-pip
$ sudo apt-get install python3-venv
```

On a fresh Ubuntu install, you need to run the following before the above commands:

``` text
$ sudo apt-get update
```

The dependent packages can either be installed automatically by an automated script or manually.

**Automated script**

The `<install-dir>/target-linux-x64/python/packages/nsys_recipe/install.py` script automates the installation of the recipe dependencies. You must select one of the following options when you run the script: `--current`, `--venv`` ``PATH`, or `--download`. Additionally, use `--offline` with `--current` or `--venv` to install from previously downloaded packages.

Options:

- `-h`: Display help

- `--current`: Install packages in the current environment. If a venv is active, packages will be installed there. Otherwise, packages will be installed in the system site-packages directory. It enables usage of `nsys`` ``recipe` without having to source a virtual environment. However, new packages risk colliding with existing ones if different versions are required.

- `--venv`` ``PATH`: Install packages in a virtual environment. If it doesn’t already exist, it is created. It prevents risk of package collision in the current environment but requires the virtual environment to be activated before running `nsys`` ``recipe`.

- `--download`: Download wheel packages for offline installation

- `--offline`: Install packages from downloaded wheels (offline mode)

- `--no-jupyter`: Do not install requirements for the jupyter notebook

- `--no-dask`: Do not install requirements for Dask

- `--quiet`: Only display errors

**Manual steps**

If you would rather install the dependencies manually, please follow the following steps:

- Create a virtual environment.

  We recommend creating a virtual environment to avoid installing packages directly into your system Python. The commands create the virtual environment in the current working directory.

  See [venv - python doc](https://docs.python.org/3/library/venv.html).

  To create a venv named recipe_env:

  ``` text
  $ python3 -m venv recipe_env
  $ source recipe_env/bin/activate
  ```

- List of dependencies.

  We have three files located in `<install-dir>/target-linux-x64/python/packages/nsys_recipe/requirements` for the dependencies:

  - Common.txt (required): dependencies needed by all recipes.

  - Dask.txt (optional): dependencies needed by the Dask mode.

  - Jupyter.txt (optional): dependencies needed to open the Jupyter notebook.

**One-step installation**

The following command will install all dependencies for CLI and GUI. Please note that you will want to activate your venv first as described above, otherwise the modules will not be available in the venv.

``` text
$ python3 -m pip install -r nsys_recipe/requirements/dask.txt -r nsys_recipe/requirements/common.txt -r nsys_recipe/requirements/jupyter.txt
```

**Two-step installation (for machines without internet)**

If you wish to download the dependencies on a machine without internet, you can download the wheel packages on a machine with internet, transfer them to the target machine and install the packages there.

On the machine with internet:

``` text
$ python3 -m pip download -r nsys_recipe/requirements/dask.txt -r nsys_recipe/requirements/common.txt -r nsys_recipe/requirements/jupyter.txt -d recipe-deps
$ tar -cvzf recipe-deps.tar.gz recipe-deps
```

On the machine with no internet:

``` text
$ tar -xvzf recipe-deps.tar.gz
$ python3 -m pip install recipe-deps/*.whl --no-index
```

**Jupyter Notebook**

The Nsight Systems UI has the ability to internally load a Jupyter notebook. It uses the Jupyter notebook installation associated with the Python on your \$PATH, which is expected to be the Python installed into the virtual environment created in the earlier steps of this guide.

If Jupyter is installed in a different location, you can add a third variable to the config.ini file that will override the default path to Jupyter:

``` text
JupyterPythonExe="/path/to/recipe_env/bin/python"
```

This config.ini file should be placed in `<install_dir>/host-linux-x64`

Note that on Windows, the path should use Windows slashes and they must be double slashes:

``` text
JupyterPythonExe="c:\\path\\to\\recipe_env\\bin\\python.exe"
```

Alternatively, you may launch Jupyter Lab independently to view the recipe outside of the Nsight Systems UI.

Ensure that the appropriate virtual environment is activated first if you have configured one, then navigate to the recipe directory and run:

``` text
jupyter-lab
```
