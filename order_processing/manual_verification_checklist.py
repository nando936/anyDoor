"""
CRITICAL MANUAL VERIFICATION CHECKLIST
This is a REDUNDANT check - completely separate from the door list generation
Must be done MANUALLY by re-reading the original PDF/order from scratch
"""

import sys
import os
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def create_manual_verification_checklist(order_name):
    """
    Creates a manual verification checklist that forces line-by-line comparison
    """
    
    checklist = f"""
================================================================================
CRITICAL MANUAL VERIFICATION CHECKLIST
Order: {order_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

[!] THIS IS A REDUNDANT CHECK - START FROM SCRATCH
[!] DO NOT TRUST ANY PREVIOUS PROCESSING
[!] MISTAKES ARE EXTREMELY EXPENSIVE TO FIX AFTER PRODUCTION

STEP 1: OPEN THE ORIGINAL PDF/ORDER
----------------------------------------
[ ] Open the original user-submitted order in "need to process" folder
[ ] Open the generated finish door list HTML in "output" folder
[ ] Place them side by side on your screen

STEP 2: VERIFY ORDER HEADER INFORMATION
----------------------------------------
[ ] Customer Name matches exactly
[ ] Job Name/Number matches exactly
[ ] Date matches
[ ] Door Style number matches (e.g., #231)
[ ] Wood Species matches (check if mixed species)

STEP 3: CRITICAL - CHECK OPENING vs FINISH SIZES
----------------------------------------
[ ] Look at original order - find the checkbox for door sizes:
    [ ] If "Opening Sizes" is checked - YOU MUST ADD 2 x overlay
    [ ] If "Finish Sizes" is checked - sizes should match exactly
    
[ ] What is the overlay specified? _____________
[ ] Calculate: 2 x overlay = _____________
[ ] This amount must be added to BOTH width AND height if Opening Sizes

STEP 4: LINE-BY-LINE VERIFICATION
----------------------------------------
Go through EACH line of the original order and check:

LINE #1:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #2:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #3:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #4:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #5:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #6:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #7:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

LINE #8:
[ ] Quantity: Original _____ vs Door List _____
[ ] Width: Original _____ 
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Height: Original _____
    If Opening Size, add _____ = Expected Finish _____
    Door List shows: _____
[ ] Type (door/drawer): Original _____ vs Door List _____
[ ] Material/Species: Original _____ vs Door List _____
[ ] Special Notes: _________________________________
    [ ] Verified notes transferred correctly

(Add more lines as needed)

STEP 5: CRITICAL NOTES CHECK
----------------------------------------
[ ] Check for "NO BORE" or "NO HINGE BORING" instructions
    Lines with no bore: _________________________________
    [ ] Verified these show in door list notes
    
[ ] Check for trash drawer pulls
    Lines with trash pulls: _____________________________
    [ ] Verified these show in door list notes
    
[ ] Check for any special instructions in original
    Special instructions: ________________________________
    [ ] Verified ALL transferred to door list

STEP 6: TOTAL COUNT VERIFICATION
----------------------------------------
[ ] Count total doors in original: _____
[ ] Count total doors in door list: _____
[ ] THESE MUST MATCH EXACTLY

[ ] If mixed species, verify counts by species:
    Paint Grade: Original _____ vs Door List _____
    White Oak: Original _____ vs Door List _____
    Other: Original _____ vs Door List _____

STEP 7: FINAL SIGN-OFF
----------------------------------------
[ ] I have manually verified EVERY line item
[ ] I have double-checked the size conversions if Opening Sizes
[ ] I have verified all special notes transferred
[ ] I have counted total quantities and they match
[ ] I understand mistakes are EXTREMELY EXPENSIVE after production

Verified by: _________________________________
Date: _______________________________________
Time spent on verification: __________________

ERRORS FOUND (if any):
_________________________________________________
_________________________________________________
_________________________________________________
_________________________________________________

CORRECTIVE ACTIONS TAKEN:
_________________________________________________
_________________________________________________
_________________________________________________
_________________________________________________

================================================================================
"""
    
    return checklist

def save_verification_checklist(order_name, output_path=None):
    """
    Save the verification checklist to a file
    """
    checklist = create_manual_verification_checklist(order_name)
    
    if not output_path:
        output_path = f"manual_verification_{order_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(checklist)
    
    print(f"[OK] Manual verification checklist created: {output_path}")
    print("\n[!] INSTRUCTIONS:")
    print("1. Print this checklist")
    print("2. Open the original order PDF")
    print("3. Open the generated door list HTML")
    print("4. Go through EVERY checkbox manually")
    print("5. Do NOT skip any steps")
    print("6. This redundant check prevents expensive mistakes")
    
    return output_path

if __name__ == "__main__":
    # Example usage
    order_name = "paul_revere_231"
    
    print("=" * 60)
    print("CREATING MANUAL VERIFICATION CHECKLIST")
    print("=" * 60)
    print("\nThis creates a checklist for MANUAL line-by-line verification")
    print("This is a REDUNDANT check - completely separate from automated processing")
    print("\n[!] This verification MUST be done after EVERY door list creation")
    
    checklist_file = save_verification_checklist(order_name)
    
    print(f"\n[!] Now open {checklist_file} and complete the verification")
    print("[!] Do not proceed to production until this checklist is complete")