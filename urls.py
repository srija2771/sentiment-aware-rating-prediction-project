from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Signup', views.Signup, name="Signup"),
	       path('SignupAction', views.SignupAction, name="SignupAction"),
	       path('LoadDataset', views.LoadDataset, name="LoadDataset"),
	       path('LoadDatasetAction', views.LoadDatasetAction, name="LoadDatasetAction"),
	       path('TrainCNN', views.TrainCNN, name="TrainCNN"),
	       path('FileComment', views.FileComment, name="FileComment"),
	       path('FileCommentAction', views.FileCommentAction, name="FileCommentAction"),	
	       path('SingleComment', views.SingleComment, name="SingleComment"),
	       path('SingleCommentAction', views.SingleCommentAction, name="SingleCommentAction"),
	       
]