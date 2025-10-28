# app/api/routes_subscriptions.py

import os
import json
import hmac
import hashlib
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
import datetime
from paystackapi.paystack import Paystack

# --- Local Imports ---
from app.services.database import get_db
from app.models.users import User
from app.subscription_model import SubscriptionTier
from app.core.dependencies import get_current_db_user

# Import the key from your config file
from app.core.config import PAYSTACK_SECRET_KEY

# --- Initialize Paystack API ---
if not PAYSTACK_SECRET_KEY:
    raise RuntimeError("PAYSTACK_SECRET_KEY not set in environment")

# Initialize the paystackapi with your secret key
paystack_instance = Paystack(secret_key=PAYSTACK_SECRET_KEY)
router = APIRouter(prefix="/subscriptions", tags=["Subscriptions"])


# --- Pydantic Response Model ---
class CheckoutResponse(BaseModel):
    authorization_url: str


class SubscriptionStatusResponse(BaseModel):
    subscription_tier: SubscriptionTier
    is_active: bool
    expires_at: datetime.datetime | None


# -----------------------------------------------
# ENDPOINT 1: CREATE CHECKOUT SESSION
# -----------------------------------------------
@router.post("/create-checkout", response_model=CheckoutResponse)
def create_checkout_session(user: User = Depends(get_current_db_user)):
    """
    Creates a Paystack checkout session for the logged-in user.
    """
    # Price is NGN 5,999. Paystack requires it in kobo.
    amount_kobo = 3499 * 100

    try:
        # Create a new transaction
        response = paystack_instance.transaction.initialize(
            email=user.email,
            amount=amount_kobo,
            # We pass the user_id in metadata. This is CRITICAL.
            # It's how we know who to upgrade when the webhook fires.
            metadata={"user_id": user.id, "username": user.username},
            currency="NGN",
            # You can add a callback_url to redirect to your frontend
            # callback_url="http://localhost:5173/payment-success"
        )

        # Check if Paystack returned a successful response
        if response["status"] == True:
            auth_url = response["data"]["authorization_url"]
            return CheckoutResponse(authorization_url=auth_url)
        else:
            # Log this! Paystack returned an error.
            print(f"Paystack error: {response['message']}")
            raise HTTPException(
                status_code=500, detail="Payment gateway error: " + response["message"]
            )

    except Exception as e:
        print(f"Error creating checkout: {e}")
        raise HTTPException(status_code=500, detail=f"Payment gateway error: {str(e)}")


# -----------------------------------------------
# ENDPOINT 2: PAYSTACK WEBHOOK
# -----------------------------------------------
@router.post("/webhook/paystack")
async def paystack_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handles incoming webhooks from Paystack to confirm payments.
    This endpoint is called by PAYSTACK'S SERVER, not by your user.
    """
    # Get the raw request body
    payload = await request.body()

    # Get the signature from the request header
    paystack_signature = request.headers.get("x-paystack-signature")
    if not paystack_signature:
        raise HTTPException(
            status_code=400, detail="Missing 'x-paystack-signature' header"
        )

    # --- 1. Verify the webhook signature (CRITICAL for security) ---
    hash_obj = hmac.new(
        PAYSTACK_SECRET_KEY.encode("utf-8"), payload, hashlib.sha512
    ).hexdigest()

    if hash_obj != paystack_signature:
        # The request is not from Paystack. REJECT IT.
        raise HTTPException(status_code=400, detail="Invalid signature")

    # --- 2. Parse the event ---
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event.get("event")

    # --- 3. Handle the "charge.success" event ---
    if event_type == "charge.success":
        data = event.get("data")

        # Get the user_id we stored in metadata
        user_id = data.get("metadata", {}).get("user_id")

        if not user_id:
            # Log this! A payment was made without a user_id.
            print("Webhook Error: 'user_id' not found in metadata")
            return {"status": "error", "message": "Missing user_id in metadata"}

        # --- 4. Find the user in our database ---
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            # Log this! User paid but doesn't exist in our DB.
            print(f"Webhook Error: User with ID {user_id} not found")
            return {"status": "error", "message": "User not found"}

        # --- 5. UPGRADE THE USER! ---
        # This is the whole point!
        user.subscription_tier = SubscriptionTier.PRO
        # Set expiry to 30 days from now
        user.subscription_expires_at = datetime.datetime.now(
            datetime.UTC
        ) + datetime.timedelta(days=30)

        db.commit()

        print(
            f"SUCCESS: User {user.username} (ID: {user.id}) has been upgraded to PRO."
        )
        # Optional: Send a "Welcome to Pro" email here.

    return {"status": "success"}


# -----------------------------------------------
# ENDPOINT 3: GET CURRENT SUBSCRIPTION STATUS
# -----------------------------------------------
@router.get("/status", response_model=SubscriptionStatusResponse)
def get_subscription_status(user: User = Depends(get_current_db_user)):
    """
    Gets the current logged-in user's subscription status.
    """
    is_active = False

    # Check if user is PRO
    if user.subscription_tier == SubscriptionTier.PRO:
        # If they are PRO, check if their subscription is active
        if (
            user.subscription_expires_at
            and user.subscription_expires_at > datetime.datetime.now(datetime.UTC)
        ):
            is_active = True
        else:
            # Their subscription has expired
            is_active = False
            # You could optionally downgrade them here, but it's
            # better to do this with a scheduled job or when they log in.
            # For now, just reporting is_active = False is fine.

    return SubscriptionStatusResponse(
        subscription_tier=user.subscription_tier,
        is_active=is_active,
        expires_at=user.subscription_expires_at,
    )
