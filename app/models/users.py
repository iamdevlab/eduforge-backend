# app/models/users.py


import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum  # <-- Added Enum
from sqlalchemy.orm import relationship  # <-- Added relationship
from app.services.database import Base

# Import the new Enum from your new file
from app.subscription_model import SubscriptionTier


class User(Base):
    """
    SQLAlchemy User Model

    Represents the 'users' table in the database.
    """

    __tablename__ = "users"

    # --- Your Existing Fields ---
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # --- NEW FIELDS FOR SUBSCRIPTIONS ---

    # The user's current subscription plan
    subscription_tier = Column(
        Enum(SubscriptionTier), nullable=False, default=SubscriptionTier.FREE
    )

    # When the current 'pro' plan expires
    subscription_expires_at = Column(DateTime, nullable=True)

    # --- NEW RELATIONSHIP ---

    # This links the User to their usage limits
    # It's a one-to-one relationship (uselist=False)
    usage = relationship(
        "UsageLimits",
        back_populates="owner",
        uselist=False,
        cascade="all, delete-orphan",
    )


# import datetime
# from sqlalchemy import Column, Integer, String, DateTime
# from app.services.database import Base

# class User(Base):
#     """
#     SQLAlchemy User Model

#     Represents the 'users' table in the database.
#     """
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True, nullable=False) # ADD THIS LINE
#     email = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     created_at = Column(DateTime, default=datetime.datetime.utcnow)
