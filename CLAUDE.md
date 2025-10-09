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

## Auto-Shutdown System
**Purpose**: Automatically hibernate VPS after 15 minutes idle to save resources while preserving full system state

**Files**:
- Script: `/home/debian/auto-shutdown.sh`
- README: `/home/debian/AUTO_SHUTDOWN_README.md`
- Runs via: root crontab (every minute)

**Key Points**:
- Hibernates after 15 min keyboard/mouse inactivity
- Preserves all applications and sessions
- Uses "shutdown" hibernate mode for VM compatibility
- Auto-clears stale state on fresh boot
- See README for configuration, testing, and troubleshooting

**Quick Commands**:
```bash
# Check if running
sudo crontab -l | grep auto-shutdown

# View logs
sudo journalctl -f | grep -i "auto-shutdown"

# Check idle status
cat /run/autoshutdown/idle_since 2>/dev/null

# Test hibernate (saves work first!)
sudo systemctl hibernate
```

## Inhouse Door Order - Debugging Measurement Classifications

**Location**: `raised panel door OS/Process Incoming Door Orders/Inhouse Door Order/`

### Debug Log Analysis Protocol

**When asked "why was measurement X classified as WIDTH/HEIGHT/UNCLASSIFIED?":**

Provide ONLY the actual data from the `*_debug.txt` file for that measurement:

1. **For each rotation attempt (0°, +22.5°, -22.5°):**
   - What lines were detected in each ROI (Left H-ROI, Right H-ROI, Top V-ROI, Bottom V-ROI)
   - The exact angles found for each line
   - How many lines passed/failed filtering (with reasons)
   - Whether arrows were detected

2. **The exact logic that led to classification:**
   - Which ROIs had indicators (L-horiz, R-horiz, T-vert, B-vert)
   - The final classification decision and why

3. **NO speculation, NO "likely" responses** - only facts from the debug log

**Example Response Format:**
```
M11 '18 1/16' at (630, 2000):

Attempt 1 (0° rotation):
- Left H-ROI: 0 lines
- Right H-ROI: 1 line (angle=86.3°)
- Top V-ROI: 5 lines
- Bottom V-ROI: 9 lines (all filtered - angles 0-38°, not in 55-125° range)
- Result: UNCLASSIFIED (L-horiz:False R-horiz:False)

Attempt 2 (+22.5° rotation):
- Top V-ROI: 8 lines, up-arrow=True
- Result: UNCLASSIFIED (T-vert:True B-vert:False)

Attempt 3 (-22.5° rotation):
- Left H-ROI: 0 lines
- Right H-ROI: 1 line (angle=-13.8°)
- Top V-ROI: 1 v-line (angle=-69.1°, adjusted=-91.6°)
- Bottom V-ROI: 1 v-line (angle=-68.6°, adjusted=-91.1°)
- Result: "Found vertical indicators: top(1 v-lines) + bottom(1 v-lines) → HEIGHT"
- Final: HEIGHT (via -22.5° rotation)
```

**Debug logs are automatically saved as**: `page_X_debug.txt` in the same directory as the input image.

## Session Startup
1. Check comments.txt for any pending tasks
2. Review project status
3. Continue with development