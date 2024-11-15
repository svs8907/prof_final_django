from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.urls import path
from .views import logoutPage
from .views import appointments


urlpatterns = [
    path('', views.login_view, name='login'),  # default URL is the login page
    path('home', views.home_view, name='home'),
    path('add', views.add, name='add'),
    path('logout/',logoutPage, name='logout'),
    path('rep/', views.rep_page, name='rep'),
    path('rep/', views.rep_view, name='rep'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('features/',views.features,name='features'),
    path('signup/', views.signup, name='signup'),
    path('find_hospitals/', views.find_hospitals, name='find_hospitals'),
    path('find_doctors/', views.find_doctors, name='find_doctors'),
    path('best_countries/', views.best_countries, name='best_countries'),
    path('register/',views.registerPage, name='register'),
    path('appointments/',appointments,name='appointments')   
]