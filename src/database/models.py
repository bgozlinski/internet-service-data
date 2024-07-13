from sqlalchemy import Column, Integer, Boolean, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Prediction(Base):
    __tablename__ = 'prediction'
    id_pred = Column(Integer, primary_key=True, index=True)
    is_tv_subscriber_pred = Column(Boolean)
    is_movie_package_subscriber_pred = Column(Boolean)
    subscription_age_pred = Column(Float)
    bill_avg_pred = Column(Integer)
    reamining_contract_pred = Column(Float)
    contracted_pred = Column(Integer)
    service_failure_count_pred = Column(Integer)
    download_avg_pred = Column(Float)
    upload_avg_pred = Column(Float)
    download_over_limit_pred = Column(Integer)
    subscription_type_Streaming_pred = Column(Boolean)
    subscription_type_TV_pred = Column(Boolean)
    subscription_type_TV_and_Streaming_pred = Column(Boolean)
    prediction_prob = Column(Float)
    model_used = Column(String)

    def to_dict(self) -> dict:
        return {
            'id_pred': self.id_pred,
            'is_tv_subscriber_pred': self.is_tv_subscriber_pred,
            'is_movie_package_subscriber_pred': self.is_movie_package_subscriber_pred,
            'subscription_age_pred': self.subscription_age_pred,
            'bill_avg_pred': self.bill_avg_pred,
            'reamining_contract_pred': self.reamining_contract_pred,
            'contracted_pred': self.contracted_pred,
            'service_failure_count_pred': self.service_failure_count_pred,
            'download_avg_pred': self.download_avg_pred,
            'upload_avg_pred': self.upload_avg_pred,
            'download_over_limit_pred': self.download_over_limit_pred,
            'subscription_type_Streaming_pred': self.subscription_type_Streaming_pred,
            'subscription_type_TV_pred': self.subscription_type_TV_pred,
            'subscription_type_TV_and_Streaming_pred': self.subscription_type_TV_and_Streaming_pred,
            'prediction_prob': self.prediction_prob,
            'model_used': self.model_used

        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            is_tv_subscriber_pred=data.get('is_tv_subscriber_pred'),
            is_movie_package_subscriber_pred=data.get('is_movie_package_subscriber_pred'),
            subscription_age_pred=data.get('subscription_age_pred'),
            bill_avg_pred=data.get('bill_avg_pred'),
            reamining_contract_pred=data.get('reamining_contract_pred'),
            contracted_pred=data.get('contracted_pred'),
            service_failure_count_pred=data.get('service_failure_count_pred'),
            download_avg_pred=data.get('download_avg_pred'),
            upload_avg_pred=data.get('upload_avg_pred'),
            download_over_limit_pred=data.get('download_over_limit_pred'),
            subscription_type_Streaming_pred=data.get('subscription_type_Streaming_pred'),
            subscription_type_TV_pred=data.get('subscription_type_TV_pred'),
            subscription_type_TV_and_Streaming_pred=data.get('subscription_type_TV_and_Streaming_pred'),
            prediction_prob=data.get('prediction_prob'),
            model_used=data.get('model_used')
        )
