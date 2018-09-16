Hackathon::Application.routes.draw do
  root to: 'main#index'
  get '/login', to: 'main#login', as: "login"
  get '/demo', to: 'demo#new', as: "demo"
  post '/demo', to: 'demo#create', as: "demo"
  get '/success', to: 'demo#success', as: "demo_success"
  get '/failure', to: 'demo#failure', as: "demo_success"
end
