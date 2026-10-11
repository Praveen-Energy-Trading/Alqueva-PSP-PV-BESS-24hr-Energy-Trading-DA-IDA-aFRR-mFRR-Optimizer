<#
.SYNOPSIS
  Registers the two daily runs of the data pipeline in Windows Task Scheduler.

.DESCRIPTION
  Run 1 (morning)   : yesterday's final files (continuous market, REN prices) and today's intraday auctions.
  Run 2 (afternoon) : tomorrow's day-ahead prices, published around 13:00 CET.
  Both runs use the same command; days already stored are skipped, so running twice is cheap.

  Without -Register the script only PRINTS what it would create (nothing on the system changes).
  Times are LOCAL times of this computer. CET = Lisbon time + 1 hour.

.EXAMPLE
  .\register_windows_tasks.ps1                       # show the plan
  .\register_windows_tasks.ps1 -Register             # create the tasks (07:30 and 13:30 local)
  .\register_windows_tasks.ps1 -Register -Morning 06:30 -Afternoon 12:30
  Unregister-ScheduledTask -TaskName 'Alqueva data pipeline (morning)','Alqueva data pipeline (afternoon)'
#>
param(
    [string]$Python    = "python",
    [string]$Morning   = "07:30",
    [string]$Afternoon = "13:30",
    [switch]$Register
)

$repo   = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script = Join-Path $repo "phase_0_data_pipeline\run_data_pipeline.py"
$tasks  = @(
    @{ Name = "Alqueva data pipeline (morning)";   At = $Morning },
    @{ Name = "Alqueva data pipeline (afternoon)"; At = $Afternoon }
)

foreach ($t in $tasks) {
    Write-Host ("{0,-40} daily at {1}  ->  {2} `"{3}`"  (working dir {4})" -f $t.Name, $t.At, $Python, $script, $repo)
    if ($Register) {
        $action  = New-ScheduledTaskAction -Execute $Python -Argument "`"$script`"" -WorkingDirectory $repo
        $trigger = New-ScheduledTaskTrigger -Daily -At $t.At
        $set     = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
                       -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger -Settings $set -Force | Out-Null
        Write-Host "  registered."
    }
}
if (-not $Register) { Write-Host "`nNothing registered. Add -Register to create the tasks." }
