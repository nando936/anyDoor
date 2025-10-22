# AnyDoor Project Instructions

## Project Overview
AnyDoor - Door monitoring and control project

## Order Processing Instructions

### WHEN USER SAYS "PROCESS ORDER":
1. **FORGET EVERYTHING** - Act as if you've NEVER seen these files before
2. **NO "I ALREADY..."** - Never say "I already ran" or "I already extracted"
3. **START AT STEP 1** - Even if you think you did it before
4. **READ** `order_processing/README.md` - This is the GOVERNING DOCUMENT
5. **FOLLOW** those instructions from the beginning

### KEY REMINDERS (Full details in README.md):
- **Manual extraction** = Claude reads PDF and fills template (not scripts)
- **No new scripts** = Use existing tools only
- **Start fresh** = Never use old data or templates
- **Three-step rule**: Extract → Process → Verify

### ⚠️ RUSH CHECK:
**SLOW IS FAST** - Following README.md correctly first time > restarting 3 times
- Creating scripts? STOP - you're rushing
- Automating extraction? STOP - you're rushing  
- Skipping steps? STOP - you're rushing

### 📍 WHERE TO UPDATE WHEN ISSUES ARISE:
- **Order processing workflow issues** → Update `order_processing/README.md`
- **Claude-specific behavior issues** → Update this file (CLAUDE.md)
- **Technical/code issues** → Fix the actual scripts

**GOVERNING DOCUMENT:** `order_processing/README.md` - Always defer to it for order processing

## Quick Commands
- `*` - Check comments.txt for tasks and process them

## Comments.txt File Rules
**LOCATION**: comments.txt is ALWAYS in the PROJECT ROOT FOLDER
- For anydoor project: C:\Users\nando\Projects\anydoor\comments.txt

See `comments_text_file__execution_guide.txt` in project root for all comments.txt instructions and usage.

**IMPORTANT**: When handling comments:
- Comments from webpage should be marked as: ### nando Comment (Webpage) - HH:MM
- Comments from CLI should be marked as: ### nando Comment (CLI) - HH:MM
- Claude responses should be marked as: ### Claude Response (CLI)
- Follow the execution guide for processing and responding to comments
- **ONLY respond to comments that are already in comments.txt** - Do not post chat conversations to comments.txt
- When using the `*` command, read and respond to existing comments in the file

**CRITICAL - NO SPECIAL CHARACTERS IN COMMENTS**:
- **NEVER use emojis** (no emojis of any kind)
- **Use plain text only** - no special Unicode characters
- **Use standard ASCII characters** for formatting:
  - Use [x] instead of checkmarks
  - Use [!] instead of warning symbols
  - Use * or - for bullet points
  - Use ** for bold text
- This ensures comments display properly on the webpage without JavaScript errors

**CRITICAL - 40 CHARACTER WIDTH**:
- **Max 40 characters per line**
- Comments are read on phone screens
- Break long lines into multiple lines
- Format lists to be narrow
- Keep text compact for mobile viewing

## Inhouse Door Order - Debug Text Files

**Location**: `raised panel door OS/Process Incoming Door Orders/Inhouse Door Order/`

### What is the Debug Text File?

**File naming**: `page_X_debug.txt` (e.g., `page_8_debug.txt`)
**Location**: Same directory as the input image
**Purpose**: Complete log of ALL processing phases for that page

**ALWAYS check the debug.txt file BEFORE running the page again when answering questions!**

### What the Debug File Contains

The debug.txt file captures the **complete stdout** from processing a page. It includes:

**PHASE 1: Finding Interest Areas**
- HSV preprocessing results
- Overlay info detection (e.g., "11/16 OL")
- Room name detection (e.g., "Bath #4")
- All text items found by Google Vision OCR
- Grouping and merging of text areas

**PHASE 2: Zoom Verification**
- For each measurement candidate
- OCR re-verification at 3x zoom
- Calculated text bounds (left, right, top, bottom)
- Text width and height in pixels
- Whether measurement is finished size (F) or opening size

**PHASE 3: Classification (WIDTH/HEIGHT/UNCLASSIFIED)**
- For each verified measurement:
  - Text position and bounds
  - H-ROI and V-ROI coordinates
  - Lines detected in each ROI (with angles)
  - Arrow detection results
  - **Rotation attempts** (0°, ±22.5°, ±30°, ±45°):
    - Lines found at each rotation
    - Scoring: `Found h-lines at angle=X°, deviation=Y° (score=Z)`
    - **BEST ROTATION** selection
  - Final classification decision (WIDTH/HEIGHT/UNCLASSIFIED)
  - Extent calculation results

**PHASE 4: Pairing Measurements into Openings**
- Down arrow scan results
- Width-height pairing logic
- Drawer configuration detection
- Bottom width classification
- Final opening specifications

**PHASE 5: Visualization**
- Visualization toggle status
- Files saved

### Debug Log Analysis Protocol

**When asked ANY question about a page (e.g., "why was M11 classified as width?", "what was the score at -30°?"):**

1. **ALWAYS check `page_X_debug.txt` FIRST**
2. **DO NOT re-run the page** unless the debug file is missing or outdated
3. **Provide ONLY facts from the debug log** - NO speculation, NO "likely" responses
4. **Quote exact output** from the debug file

**Example: Answering rotation scoring questions:**
```bash
grep -A 100 "Analyzing measurement 11:" page_8_debug.txt | grep -E "score=|BEST ROTATION"
```

**Example Response Format:**
```
M11 '18 1/16' at (642, 2000):

Rotation scores from debug.txt:
- -22.5°: score=8.0 (h-lines at angle=-14.5°, deviation=8.0°) ← BEST
- -30.0°: score=25.2 (h-lines at angle=-4.8°, deviation=25.2°)

M11 chose -22.5° because it had the best (lowest) score.
```

## Using Sub-Agents

### Explore Sub-Agent (For Questions)

**IMPORTANT**: When the user asks questions about code or debug files, use the Explore sub-agent instead of running search commands directly.

**Use Explore agent for:**
- Code structure questions (e.g., "how does rotation scoring work?", "where is arrow detection implemented?")
- Debug file analysis (e.g., "why was M11 classified as width?", "what scores did M4 get?")
- Log file questions
- Codebase exploration ("where are errors handled?", "how does pairing work?")

**Example:**
```
User: "How does the rotation scoring algorithm work?"
Claude: [Uses Task tool with subagent_type=Explore to search the codebase]
```

**Benefits:**
- More efficient search
- Better context usage
- Faster responses
- More thorough exploration

**Do NOT use Explore for:**
- Single file reads (use Read tool instead)
- Specific file path searches (use Glob tool instead)

### General-Purpose Agent (For Code Changes)

**IMPORTANT**: When the user requests code changes, use the general-purpose agent to make the changes.

**Use general-purpose agent for:**
- Any code modifications requested by user
- Multi-step code refactoring
- Feature implementation
- Bug fixes
- Code updates based on findings from Explore agent

**Example:**
```
User: "Change the rotation angle threshold from 8.0 to 15.0"
Claude: [Uses Task tool with subagent_type=general-purpose to make the change]
```

**Benefits:**
- Handles complex multi-file changes
- Can search, analyze, and edit in one workflow
- Better tracking of changes made
- Reduces context usage in main conversation

## Work Time Tracker

### System Overview
The user has a work time tracking system that monitors keyboard/mouse activity and logs work sessions. The tracker is displayed in Conky (desktop widget) showing daily work time.

### Key Locations
- **Tracker script**: `/home/nando/.local/bin/work-time-tracker`
- **Log directory**: `/home/nando/.local/share/time-tracker/`
- **Daily logs**: `work-time-YYYY-MM-DD.log` (e.g., `work-time-2025-10-15.log`)
- **State file**: `tracker-state` (tracks if session is active)
- **Conky script**: `/home/nando/.local/bin/work-time-conky`

### How It Works
1. **work-time-tracker** runs continuously as a background process
2. Monitors idle time using `xprintidle`
3. After 8 minutes of inactivity, marks session as ended
4. Logs SESSION_START and SESSION_END to daily log file
5. Log file path is calculated dynamically each time log_message() is called

### Conky Display Format
`D4: 5h 2m (1h 21m)`
- **D4** = Day 4 of tracking (number of log files)
- **5h 2m** = Total work time today
- **(1h 21m)** = Current active session duration

### Common Issues

#### "Work time shows only 1h but I've been working all day"
**Problem**: Tracker started days ago and never restarted, so log file path was set once at startup and never updated for new days.

**Solution**:
1. Check tracker start time: `ps -o lstart,cmd -p $(pgrep -f work-time-tracker)`
2. If running for multiple days, restart: `pkill -f work-time-tracker && nohup /home/nando/.local/bin/work-time-tracker > /dev/null 2>&1 &`
3. Move misplaced entries to correct date file using grep

**Prevention**: The tracker script now calculates log file path dynamically in log_message() function (fixed 2025-10-15)

#### How to Check Work Time
```bash
# Current time via conky
/home/nando/.local/bin/work-time-conky

# View today's log
cat ~/.local/share/time-tracker/work-time-$(date +%Y-%m-%d).log

# Check tracker status
cat ~/.local/share/time-tracker/tracker-state

# Check if tracker is running
ps aux | grep work-time-tracker | grep -v grep
```

### When User Mentions "work time" or "conky"
They are referring to this time tracking system displayed in their desktop Conky widget.

## Git / GitHub

### GitHub Credentials
**Location**: `/home/nando/projects/nando936/github-token.txt`

The GitHub personal access token is stored in the nando936 project directory.

### How to Push to GitHub

**IMPORTANT**: Always use the token-based push command:

```bash
# 1. Read the token
TOKEN=$(cat /home/nando/projects/nando936/github-token.txt)

# 2. Push using token in URL
git push https://${TOKEN}@github.com/nando936/anyDoor.git master
```

**OR in one command:**
```bash
git push https://$(cat /home/nando/projects/nando936/github-token.txt)@github.com/nando936/anyDoor.git master
```

**DO NOT** use plain `git push` - it will fail with authentication error.

### Common Git Operations
- `git status` - Check current changes
- `git diff <file>` - View changes in file
- `git add <file>` - Stage file for commit
- `git commit -m "message"` - Commit with message
- `git push` - **DON'T USE** - use token-based push above

## Session Startup
1. Check comments.txt for any pending tasks
2. Review project status
3. Continue with development